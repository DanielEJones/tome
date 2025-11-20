# tome
A small, stack-based programming language in the style of forth.

# Quick Start
To run tome, run the file `tome.py` using your python interpreter and supply a file path. For example:

```console
$ python3 ./tome.py --interpret ./examples/loops.tome
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

## Compilation
Currently, tome can be compiled to target x86_64 linux by passing the `--compile` flag:

```console
$ python3 ./tome.py --compile ./examples/06_fizzbuzz.tome
$ ./out
1
2
102
4
98
102
7
8
102
98
11
102
13
14
122
16
17
102
19
```

The compiler currently relies on NASM as the assembler, so please ensure that it is installed.

# Syntax
Tome is a stack based language, meaning that all operations rely implicitly on a stack to supply their arguments and
hold their outputs.

## Literals
Tome currently supports integer and character literals:
```tome
-- push a `1` onto the stack
1           -- [ 1 ]

-- now we push a `2` and a `3` also
2 3         -- [ 1 2 3 ]

-- push the ascii representation of 'f' onto the stack
'f'         -- [ 1 2 3 102 ]
```

## Words
The main unit of computation in tome is the word. Words are like functions in regular languages, but they take their
arguments implicitly from the stack and push their output back to it when they are done.

Currently, the following list of builtin words are supported:

### Arithmetic
```tome
-- `+`
1 1 +       -- [ 2 ]

-- `-`
3 2 -       -- [ 1 ]

-- `*`
4 5 *       -- [ 20 ]

-- `/`
8 2 /       -- [ 4 ]

-- `%`
10 3 %      -- [ 1 ]
```

### Logic
Currently, tome has no concept of a boolean and relies on values of `1` and `0` for truth
```tome
-- `<`
0 1 <       -- [ 1 ]

-- `>`
0 1 >       -- [ 0 ]

-- `=`
0 1 =       -- [ 0 ]

-- `and`
0 1 and     -- [ 0 ]

-- `or`
0 1 or      -- [ 1 ]

-- `not`
1 not       -- [ 0 ]
```

### Stack Manipulation
```tome
-- `dup`
1 dup       -- [ 1 1 ]

-- `2nd`
2 1 2nd     -- [ 2 1 2 ]

-- `3rd`
3 2 1 3rd   -- [ 3 2 1 3 ]

-- `swap`
4 5 swap    -- [ 5 4 ]

-- `drop`
1 2 drop    -- [ 1 ]
```

### Memory Access
```tome
-- `#` (Pushes the address of the beggining of the heap)
#           -- [ `base` ]

-- `read` (Reads a 64 bit integer at the address specified)
# 8 + read  -- [ `heap[base+8]` ]

-- `write` (Write a 64 bit integer to the address specified)
10 # 8 + write      -- [ ]
```

### Miscellaneous
```tome
-- `.` (print)
1 2 .       -- [ 1 ]
```

## Control Flow
Tome supports the following control flow constructs:
```tome
-- select the bigger number between one and two
if 1 2 > then 1
         else 2;

-- count down from that number to zero, printing as we go
while
    dup 0 >
    2nd 0 =
    or
do
    dup . 1 - ;
```
