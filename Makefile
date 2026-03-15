CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LIBS = -lm

symnmf: symnmf.c symnmf.h
	$(CC) $(CFLAGS) symnmf.c -o symnmf $(LIBS)

clean:
	rm -f symnmf