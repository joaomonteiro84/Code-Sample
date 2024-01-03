CFLAGS = -Wall -Werror -g -Og -Wno-unused-result
CC     = gcc $(CFLAGS)
SHELL  = /bin/bash
CWD    = $(shell pwd | sed 's/.*\///g')

PROGRAMS = \
	el_malloc.o \
	el_demo \
	test_el_malloc \
	showsym \

all : $(PROGRAMS)

clean :
	rm -f $(PROGRAMS) *.o vgcore.* a.out

help :
	@echo 'Typical usage is:'
	@echo '  > make                          # build all programs'
	@echo '  > make clean                    # remove all compiled items'
	@echo '  > make zip                      # create a zip file for submission'
	@echo '  > make prob1                    # built targets associated with problem 1'
	@echo '  > make prob1 testnum=5          # run problem 1 test #5 only'
	@echo '  > make test-prob2               # run test for problem 2'
	@echo '  > make test                     # run all tests'


############################################################
# 'make zip' to create p2-code.zip for submission
AN=p5
zip : clean clean-tests
	rm -f $(AN)-code.zip
	cd .. && zip "$(CWD)/$(AN)-code.zip" -r "$(CWD)"
	@echo Zip created in $(AN)-code.zip
	@if (( $$(stat -c '%s' $(AN)-code.zip) > 10*(2**20) )); then echo "WARNING: $(AN)-code.zip seems REALLY big, check there are no abnormally large test files"; du -h $(AN)-code.zip; fi
	@if (( $$(unzip -t $(AN)-code.zip | wc -l) > 256 )); then echo "WARNING: $(AN)-code.zip has 256 or more files in it which may cause submission problems"; fi


# EL MALLOC
el_malloc.o : el_malloc.c el_malloc.h
	$(CC) -c $<

el_demo : el_demo.c el_malloc.o
	$(CC) -o $@ $^

test_el_malloc : test_el_malloc.c el_malloc.o
	$(CC) -o $@ $^

# SHOWSYM
showsym : showsym.c
	$(CC) -o $@ $^

# TESTING TARGETS
test: test-prob1 test-prob2

test-setup :
	@chmod u+x test-input/globals test-input/greet_main test-input/list_main test-input/ls test-input/naked_globals test-input/quote_main
	@chmod u+rx testy

test-prob1: showsym test-setup
	./testy test_showsym.org $(testnum)

test-prob2: test_el_malloc test-setup 
	./testy test_el_malloc.org $(testnum)

clean-tests :
	rm -rf test-results

