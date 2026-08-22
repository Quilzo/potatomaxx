# SPDX-License-Identifier: GPL-2.0-or-later
#
# GNU-conventions build wrapper around Cargo.
#
# Cargo is the real build system; this exists because distribution packagers and
# the GNU coding standards both expect ./Makefile with prefix, DESTDIR, install,
# uninstall, check and dist. See
# https://www.gnu.org/prep/standards/html_node/Makefile-Conventions.html

PACKAGE     = potatomaxx
VERSION     = 0.1.1

# GNU directory variables. Overridable, and DESTDIR is honoured everywhere.
prefix      ?= /usr/local
exec_prefix ?= $(prefix)
bindir      ?= $(exec_prefix)/bin
datarootdir ?= $(prefix)/share
datadir     ?= $(datarootdir)
mandir      ?= $(datarootdir)/man
man1dir     ?= $(mandir)/man1
infodir     ?= $(datarootdir)/info
docdir      ?= $(datarootdir)/doc/$(PACKAGE)
DESTDIR     ?=

CARGO       ?= cargo
INSTALL     ?= install
INSTALL_PROGRAM ?= $(INSTALL) -m 755
INSTALL_DATA    ?= $(INSTALL) -m 644
MAKEINFO    ?= makeinfo
GZIP        ?= gzip

CARGO_FLAGS ?= --release --locked
TARGETDIR   ?= target/release
BIN          = $(TARGETDIR)/$(PACKAGE)

.PHONY: all build check test lint fmt install uninstall clean distclean dist info html installdirs help

all: build

build:
	$(CARGO) build $(CARGO_FLAGS)

# `check` is the GNU-standard name for the test target.
# Both profiles are run: debug enables the integer overflow checks release omits,
# and this tool parses attacker-controlled files, so an overflow is a security bug.
check: test
test:
	$(CARGO) test --locked
	$(CARGO) test --locked --release

lint:
	$(CARGO) clippy --all-targets --locked -- -D warnings

fmt:
	$(CARGO) fmt --all --check

installdirs:
	$(INSTALL) -d $(DESTDIR)$(bindir) $(DESTDIR)$(man1dir) $(DESTDIR)$(docdir)

install: build installdirs
	$(INSTALL_PROGRAM) $(BIN) $(DESTDIR)$(bindir)/$(PACKAGE)
	$(INSTALL_DATA) doc/$(PACKAGE).1 $(DESTDIR)$(man1dir)/$(PACKAGE).1
	$(INSTALL_DATA) README.md CHANGELOG.md docs/KERNEL.md $(DESTDIR)$(docdir)/

uninstall:
	rm -f $(DESTDIR)$(bindir)/$(PACKAGE)
	rm -f $(DESTDIR)$(man1dir)/$(PACKAGE).1
	rm -f $(DESTDIR)$(docdir)/README.md
	rm -f $(DESTDIR)$(docdir)/CHANGELOG.md
	rm -f $(DESTDIR)$(docdir)/KERNEL.md
	-rmdir $(DESTDIR)$(docdir)

# Texinfo manual, as the GNU standards require.
info: doc/$(PACKAGE).info
doc/$(PACKAGE).info: doc/$(PACKAGE).texi
	$(MAKEINFO) --output=$@ $<

html: doc/$(PACKAGE).texi
	$(MAKEINFO) --html --no-split --output=doc/$(PACKAGE).html $<

clean:
	$(CARGO) clean
	rm -f doc/$(PACKAGE).info doc/$(PACKAGE).html

distclean: clean
	rm -f $(PACKAGE)-$(VERSION).tar.gz

# A release tarball, which GNU evaluation asks for.
dist:
	git archive --format=tar --prefix=$(PACKAGE)-$(VERSION)/ HEAD \
	  | $(GZIP) -9 > $(PACKAGE)-$(VERSION).tar.gz
	@echo "wrote $(PACKAGE)-$(VERSION).tar.gz"

help:
	@echo "targets: all build check lint fmt install uninstall info html clean distclean dist"
	@echo "vars:    prefix=$(prefix) bindir=$(bindir) mandir=$(mandir) DESTDIR="
