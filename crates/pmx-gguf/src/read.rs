//! GGUF header parsing.
//!
//! The header is read into memory in one shot (it is small even for very large
//! models) and then parsed with a bounds-checked cursor. Metadata values are
//! kept as raw byte spans plus their type tag, so the writer can re-emit them
//! verbatim: we never round-trip a value through a lossy representation.

use crate::{align_up, ggml::GgmlType, GgufError, DEFAULT_ALIGNMENT, MAGIC};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Ceilings that bound allocation from attacker-controlled counts.
const MAX_TENSORS: u64 = 1 << 22;
const MAX_KV: u64 = 1 << 20;
const MAX_NAME_LEN: u64 = 1 << 16;
const MAX_DIMS: u32 = 8;
/// Header region we are willing to buffer, in bytes.
const MAX_HEADER: u64 = 512 << 20;

/// A GGUF metadata value type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum ValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self, GgufError> {
        Ok(match v {
            0 => ValueType::Uint8,
            1 => ValueType::Int8,
            2 => ValueType::Uint16,
            3 => ValueType::Int16,
            4 => ValueType::Uint32,
            5 => ValueType::Int32,
            6 => ValueType::Float32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::Uint64,
            11 => ValueType::Int64,
            12 => ValueType::Float64,
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }

    fn tag(self) -> u32 {
        match self {
            ValueType::Uint8 => 0,
            ValueType::Int8 => 1,
            ValueType::Uint16 => 2,
            ValueType::Int16 => 3,
            ValueType::Uint32 => 4,
            ValueType::Int32 => 5,
            ValueType::Float32 => 6,
            ValueType::Bool => 7,
            ValueType::String => 8,
            ValueType::Array => 9,
            ValueType::Uint64 => 10,
            ValueType::Int64 => 11,
            ValueType::Float64 => 12,
        }
    }

    /// Fixed width in bytes, or `None` for variable-length types.
    fn fixed_width(self) -> Option<u64> {
        Some(match self {
            ValueType::Uint8 | ValueType::Int8 | ValueType::Bool => 1,
            ValueType::Uint16 | ValueType::Int16 => 2,
            ValueType::Uint32 | ValueType::Int32 | ValueType::Float32 => 4,
            ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => 8,
            ValueType::String | ValueType::Array => return None,
        })
    }
}

/// A metadata value, retained as its type tag plus the raw bytes it occupied.
#[derive(Debug, Clone)]
pub struct MetaValue {
    /// The value's type tag.
    pub ty: ValueType,
    /// The value's encoded bytes, exactly as they appeared in the source file.
    pub raw: Vec<u8>,
}

impl MetaValue {
    /// Interpret this value as a `u32`, if it is one.
    pub fn as_u32(&self) -> Option<u32> {
        if self.ty == ValueType::Uint32 && self.raw.len() == 4 {
            Some(u32::from_le_bytes([
                self.raw[0],
                self.raw[1],
                self.raw[2],
                self.raw[3],
            ]))
        } else {
            None
        }
    }

    /// Interpret this value as a `u64`, accepting `Uint32` widening.
    pub fn as_u64(&self) -> Option<u64> {
        match self.ty {
            ValueType::Uint32 => self.as_u32().map(u64::from),
            ValueType::Uint64 if self.raw.len() == 8 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&self.raw);
                Some(u64::from_le_bytes(b))
            }
            _ => None,
        }
    }

    /// Interpret this value as a UTF-8 string, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        if self.ty != ValueType::String || self.raw.len() < 8 {
            return None;
        }
        std::str::from_utf8(&self.raw[8..]).ok()
    }

    /// Encoded size of this value on disk.
    pub fn encoded_len(&self) -> usize {
        self.raw.len()
    }

    /// Type tag as written to disk.
    pub fn tag(&self) -> u32 {
        self.ty.tag()
    }
}

/// One tensor's header record.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name, e.g. `blk.3.ffn_up_exps.weight`.
    pub name: String,
    /// Dimensions, innermost first, as stored.
    pub dims: Vec<u64>,
    /// Element type.
    pub ggml_type: GgmlType,
    /// Byte offset of this tensor's data, relative to the start of the data section.
    pub offset: u64,
    /// Byte length of this tensor's data. Derived, not stored in the file.
    pub nbytes: u64,
}

impl TensorInfo {
    /// Total element count.
    pub fn n_elements(&self) -> Result<u64, GgufError> {
        let mut n: u64 = 1;
        for d in &self.dims {
            n = n
                .checked_mul(*d)
                .ok_or(GgufError::ArithmeticOverflow("n_elements"))?;
        }
        Ok(n)
    }
}

/// A parsed GGUF file: header in memory, tensor data left on disk.
#[derive(Debug, Clone)]
pub struct Gguf {
    /// Path this was read from.
    pub path: PathBuf,
    /// Container version.
    pub version: u32,
    /// Metadata key/value pairs, in file order.
    pub kvs: Vec<(String, MetaValue)>,
    /// Tensor records, in file order.
    pub tensors: Vec<TensorInfo>,
    /// Alignment from `general.alignment`, or the 32-byte default.
    pub alignment: u64,
    /// Absolute file offset where the tensor data section begins.
    pub data_start: u64,
    /// Total file length.
    pub file_len: u64,
}

/// Bounds-checked forward-only cursor over the buffered header.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: u64, what: &'static str) -> Result<&'a [u8], GgufError> {
        let n_us = usize::try_from(n).map_err(|_| GgufError::ArithmeticOverflow(what))?;
        let end = self
            .p
            .checked_add(n_us)
            .ok_or(GgufError::ArithmeticOverflow(what))?;
        if end > self.b.len() {
            return Err(GgufError::Truncated {
                what,
                need: n,
                have: (self.b.len() - self.p.min(self.b.len())) as u64,
            });
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, GgufError> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, GgufError> {
        let s = self.take(8, what)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(s);
        Ok(u64::from_le_bytes(b))
    }

    /// A length-prefixed UTF-8 string. Returns the string and its encoded span.
    fn string(&mut self, what: &'static str) -> Result<(String, Vec<u8>), GgufError> {
        let start = self.p;
        let len = self.u64(what)?;
        if len > MAX_NAME_LEN {
            return Err(GgufError::ImplausibleCount {
                what,
                value: len,
                limit: MAX_NAME_LEN,
            });
        }
        let bytes = self.take(len, what)?;
        let s = std::str::from_utf8(bytes)
            .map_err(|_| GgufError::BadUtf8(what))?
            .to_owned();
        Ok((s, self.b[start..self.p].to_vec()))
    }

    /// Consume one metadata value of type `ty`, returning its raw encoded bytes.
    fn value(&mut self, ty: ValueType) -> Result<Vec<u8>, GgufError> {
        let start = self.p;
        match ty {
            ValueType::String => {
                self.string("metadata string")?;
            }
            ValueType::Array => {
                let elem = ValueType::from_u32(self.u32("array element type")?)?;
                let n = self.u64("array length")?;
                match elem.fixed_width() {
                    Some(w) => {
                        let total = n
                            .checked_mul(w)
                            .ok_or(GgufError::ArithmeticOverflow("array bytes"))?;
                        self.take(total, "array data")?;
                    }
                    None => {
                        // Strings and nested arrays must be walked one by one.
                        if n > MAX_KV {
                            return Err(GgufError::ImplausibleCount {
                                what: "array length",
                                value: n,
                                limit: MAX_KV,
                            });
                        }
                        for _ in 0..n {
                            self.value(elem)?;
                        }
                    }
                }
            }
            fixed => {
                let w = fixed.fixed_width().expect("non-variable type");
                self.take(w, "metadata scalar")?;
            }
        }
        Ok(self.b[start..self.p].to_vec())
    }
}

impl Gguf {
    /// Parse the header of the GGUF file at `path`.
    ///
    /// Tensor data is not read. Every declared length is validated against the
    /// real file size before it is trusted.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref().to_path_buf();
        let mut f = File::open(&path)?;
        let file_len = f.metadata()?.len();

        // Read a bounded prefix; the header is a small fraction of any real model.
        let want = file_len.min(MAX_HEADER);
        let mut buf = vec![0u8; usize::try_from(want).unwrap_or(0)];
        {
            let mut r = BufReader::new(&mut f);
            r.read_exact(&mut buf)?;
        }

        let mut c = Cur { b: &buf, p: 0 };
        let magic = c.take(4, "magic")?;
        if magic != MAGIC {
            let mut m = [0u8; 4];
            m.copy_from_slice(magic);
            return Err(GgufError::BadMagic(m));
        }
        let version = c.u32("version")?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = c.u64("tensor count")?;
        if tensor_count > MAX_TENSORS {
            return Err(GgufError::ImplausibleCount {
                what: "tensor",
                value: tensor_count,
                limit: MAX_TENSORS,
            });
        }
        let kv_count = c.u64("kv count")?;
        if kv_count > MAX_KV {
            return Err(GgufError::ImplausibleCount {
                what: "metadata",
                value: kv_count,
                limit: MAX_KV,
            });
        }

        let mut kvs = Vec::with_capacity(usize::try_from(kv_count).unwrap_or(0).min(4096));
        for _ in 0..kv_count {
            let (key, _) = c.string("metadata key")?;
            let ty = ValueType::from_u32(c.u32("metadata value type")?)?;
            let raw = c.value(ty)?;
            kvs.push((key, MetaValue { ty, raw }));
        }

        let alignment = kvs
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .filter(|a| a.is_power_of_two() && *a > 0)
            .unwrap_or(DEFAULT_ALIGNMENT);

        let mut tensors = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0).min(65536));
        for _ in 0..tensor_count {
            let (name, _) = c.string("tensor name")?;
            let n_dims = c.u32("tensor dim count")?;
            if n_dims > MAX_DIMS {
                return Err(GgufError::ImplausibleCount {
                    what: "tensor dimension",
                    value: u64::from(n_dims),
                    limit: u64::from(MAX_DIMS),
                });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.u64("tensor dimension")?);
            }
            let ggml_type = GgmlType(c.u32("tensor type")?);
            let offset = c.u64("tensor offset")?;
            let mut ti = TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
                nbytes: 0,
            };
            ti.nbytes = ggml_type.tensor_bytes(ti.n_elements()?)?;
            tensors.push(ti);
        }

        let header_end =
            u64::try_from(c.p).map_err(|_| GgufError::ArithmeticOverflow("header end"))?;
        let data_start = align_up(header_end, alignment)
            .ok_or(GgufError::ArithmeticOverflow("data section start"))?;
        if data_start > file_len {
            return Err(GgufError::Truncated {
                what: "data section",
                need: data_start,
                have: file_len,
            });
        }
        let data_len = file_len - data_start;

        // Validate every tensor against the real data section before trusting it.
        for t in &tensors {
            if t.offset % alignment != 0 {
                return Err(GgufError::UnalignedOffset {
                    name: t.name.clone(),
                    offset: t.offset,
                    alignment,
                });
            }
            let end = t
                .offset
                .checked_add(t.nbytes)
                .ok_or(GgufError::ArithmeticOverflow("tensor end"))?;
            if end > data_len {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.offset,
                    len: t.nbytes,
                    data_len,
                });
            }
        }

        Ok(Gguf {
            path,
            version,
            kvs,
            tensors,
            alignment,
            data_start,
            file_len,
        })
    }

    /// Look up a metadata value by exact key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.kvs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The model architecture from `general.architecture`, if present.
    pub fn architecture(&self) -> Option<&str> {
        self.meta("general.architecture").and_then(|v| v.as_str())
    }

    /// Total bytes occupied by tensor data.
    pub fn data_len(&self) -> u64 {
        self.file_len.saturating_sub(self.data_start)
    }

    /// Read one slice along a tensor's last (expert) axis.
    ///
    /// Reading a whole stacked expert tensor can mean gigabytes; requantisation
    /// only ever needs one expert at a time.
    pub fn read_tensor_slice(&self, t: &TensorInfo, slice: u64) -> Result<Vec<u8>, GgufError> {
        let n_slices = *t.dims.last().unwrap_or(&1);
        if n_slices == 0 || slice >= n_slices {
            return Err(GgufError::InvalidPlan(format!(
                "slice {slice} out of range for tensor {:?} with {n_slices} slices",
                t.name
            )));
        }
        if t.nbytes % n_slices != 0 {
            return Err(GgufError::InvalidPlan(format!(
                "tensor {:?} ({} bytes) does not divide into {n_slices} slices",
                t.name, t.nbytes
            )));
        }
        let slice_bytes = t.nbytes / n_slices;
        let abs = self
            .data_start
            .checked_add(t.offset)
            .and_then(|v| v.checked_add(slice * slice_bytes))
            .ok_or(GgufError::ArithmeticOverflow("slice absolute offset"))?;
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(abs))?;
        let mut buf = vec![
            0u8;
            usize::try_from(slice_bytes)
                .map_err(|_| GgufError::ArithmeticOverflow("slice length"))?
        ];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Elements in one slice along a tensor's last axis.
    pub fn slice_elements(&self, t: &TensorInfo) -> Result<u64, GgufError> {
        let n_slices = *t.dims.last().unwrap_or(&1);
        let n = t.n_elements()?;
        if n_slices == 0 || n % n_slices != 0 {
            return Err(GgufError::InvalidPlan(format!(
                "tensor {:?} has {n} elements, not divisible by {n_slices} slices",
                t.name
            )));
        }
        Ok(n / n_slices)
    }

    /// Read one tensor's bytes from disk.
    pub fn read_tensor(&self, t: &TensorInfo) -> Result<Vec<u8>, GgufError> {
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(
            self.data_start
                .checked_add(t.offset)
                .ok_or(GgufError::ArithmeticOverflow("tensor absolute offset"))?,
        ))?;
        let mut buf = vec![
            0u8;
            usize::try_from(t.nbytes)
                .map_err(|_| GgufError::ArithmeticOverflow("tensor length"))?
        ];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }
}
