# tome
A small, stack-based programming language in the style of forth.

# Quick Start
To run tome, run the file `tome.py` using your python interpreter and supply a file path. For example:

```console
$ python3 ./tome.py ./examples/loops.tome
10
9
8
7
6
5
4
3
2
1
```

# Syntax
Tome is a stack based language, meaning that all operations rely implicitly on a stack to supply their arguments and
hold their outputs

## Literals
Tome currently supports integer literals. To push an integer literal to the stack, simply write it into your program:
```tome
-- push a `1` onto the stack
1                                   -- [ 1 ]

-- now we push a `2` and a `3` also
2 3                                 -- [ 1 2 3 ]
```

## Words
The main unit of computation in tome is the word. Words are like functions in regular languages, but they take their
arguments implicitly from the stack and push their output back to it when they are done.
```tome
1           -- [ 1 ]
2           -- [ 1 2 ]
+           -- [ 3 ]
```

Currently only the builtin words `+`, `-`, `=`, `>`, `.` and `dup` are supported.

## Stack Manipulation
To make programming easier, there are built in words to manipulate the data stack.
```tome
1           -- [ 1 ]
dup         -- [ 1 1 ]
2 dup +     -- [ 1 1 4 ]
```

## Control Flow
Tome supports the following control flow constructs:
```tome
-- select the bigger number between one and two
if 1 2 > then 1
         else 2;
         
-- count down from that number to zero, printing as we go
while dup 0 > do
    dup . 1 - ;
```