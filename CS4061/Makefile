CC = gcc
CFLAGS = -g
CLINK = -lpthread

SRCDIR=src
INCLDIR=include
LIBDIR=lib

bank: main.o producer.o consumer.o $(LIBDIR)/utils.o
	$(CC) -o $@ $^ -I$(INCLDIR) $(CFLAGS) $(CLINK)

main.o: $(SRCDIR)/main.c $(LIBDIR)/utils.o
	$(CC) -o $@ -I$(INCLDIR) -c $(SRCDIR)/main.c $(CFLAGS)

producer.o: $(SRCDIR)/producer.c $(LIBDIR)/utils.o
	$(CC) -o $@ -I$(INCLDIR) -c $(SRCDIR)/producer.c $(CFLAGS)

consumer.o: $(SRCDIR)/consumer.c $(LIBDIR)/utils.o
	$(CC) -o $@ -I$(INCLDIR) -c $(SRCDIR)/consumer.c $(CFLAGS)

$(LIBDIR)/utils.o: $(LIBDIR)/utils.c
	$(CC) -o $(LIBDIR)/utils.o -I$(INCLDIR) -c $(LIBDIR)/utils.c $(CFLAGS)

run1: bank
	./bank 1 input/t1.csv

run2: bank
	./bank 100 input/t2.csv

run3: bank
	./bank 100 input/t3.csv

run4: bank
	./bank 100 input/t4.csv

run5: bank
	./bank 100 input/t5.csv

t1: bank
	./bank 1 input/t1.csv
	diff expected/result1.txt output/result.txt
	@echo passed

t2: bank
	./bank 100 input/t2.csv
	diff expected/result2.txt output/result.txt
	@echo passed

t3: bank
	./bank 100 input/t3.csv
	diff expected/result3.txt output/result.txt
	@echo passed

t4: bank
	./bank 100 input/t4.csv
	diff expected/result4.txt output/result.txt
	@echo passed

t5: bank
	./bank 100 input/t5.csv
	diff expected/result5.txt output/result.txt
	@echo passed

test: bank
	./bank 1 input/t1.csv
	diff expected/result1.txt output/result.txt
	@echo passed
	./bank 100 input/t2.csv
	diff expected/result2.txt output/result.txt
	@echo passed
	./bank 100 input/t3.csv
	diff expected/result3.txt output/result.txt
	@echo passed
	./bank 100 input/t4.csv
	diff expected/result4.txt output/result.txt
	@echo passed
	./bank 100 input/t5.csv
	diff expected/result5.txt output/result.txt
	@echo passed

testextracredit: bank
	./bank 1 input/t1.csv -bp 200
	diff expected/result1.txt output/result.txt
	@echo passed
	./bank 100 input/t2.csv -bp 200
	diff expected/result2.txt output/result.txt
	@echo passed
	./bank 100 input/t3.csv -bp 200
	diff expected/result3.txt output/result.txt
	@echo passed
	./bank 100 input/t4.csv -bp 200
	diff expected/result4.txt output/result.txt
	@echo passed
	./bank 100 input/t5.csv -bp 200
	diff expected/result5.txt output/result.txt
	@echo passed

.PHONY: clean run1 run2 run3 run4 run5 t1 t2 t3 t4 t5 test testextracredit

clean: 
	rm bank main.o producer.o consumer.o $(LIBDIR)/utils.o
