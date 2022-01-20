#!/bin/bash

for f in $(find . -path ./build -prune -o -name '*.h' -or -name '*.c' -or -name '*.cpp' -or -name '*.cc'); do 
    if [ ${f} != './build' ]; then
        echo "Formatting ${f}"
        clang-format -i --style=file ${f}
    fi
done

echo "Done!"
