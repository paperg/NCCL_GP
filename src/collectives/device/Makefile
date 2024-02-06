
BUILDDIR ?= $(abspath ../../../build)

OBJDIR := $(BUILDDIR)/obj/collectives/device
STATICLIB  := $(OBJDIR)/colldevice.a
RULESFILE  := $(OBJDIR)/Makefile.rules

CUDA_MAJOR = 11

all: $(STATICLIB)

$(RULESFILE) : gen_rules.sh
	@printf "Generating %-35s > %s\n" rules $@
	@mkdir -p $(OBJDIR)
	@CUDA_MAJOR=${CUDA_MAJOR} ./gen_rules.sh $(OBJDIR) > $@

-include $(RULESFILE)

LIBOBJ := $(GENOBJS) 

$(STATICLIB) : $(LIBOBJ)
	echo "Archiving"
	@printf "Archiving  %-35s > %s\n" objects $@
	ar cr $@ $^

clean:
	rm -f $(RULESFILE) $(STATICLIB)