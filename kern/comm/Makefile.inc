die_collective.sass: die.sass comm/collective.lua comm/publication.sass mtp/schedule.lua mtp/prefetch.lua
	$(LUA) comm/collective.lua die.sass $@ comm/publication.sass
die_collective.cubin: die_collective.sass die_collective.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_collective.ptx
	$(MAXAS) -i die_collective.sass $@
