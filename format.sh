#!/bin/bash

for f in $(find . -name '*.h' -or -name '*.c' -or -name '*.cpp' -or -name '*.cc'); do 
    echo "Formatting ${f}"
    clang-format -i --style=file ${f}
done

echo "Done!"
