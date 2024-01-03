OBJECTS = \
	globals       \
	quote_main    \
	list_main     \
	greet_main.o  \
	greet_funcs.o \
	greet_main    \

all : $(OBJECTS)


globals : globals.c
	gcc -o $@ $^

quote_main : quote_main.c quote_data.c 
	gcc -o $@ $^

list_main : list_main.c list_funcs.c
	gcc -o $@ $^

greet_main.o : greet_main.c
	gcc -c $<

greet_funcs.o : greet_funcs.c
	gcc -c $<

greet_main : greet_main.o greet_funcs.o
	gcc -o $@ $^

# ONLY the instructor should run 'make realclean' to prevent tests
# from failing due to compiler differences
realclean:
	rm -f $(OBJECTS) a.out
