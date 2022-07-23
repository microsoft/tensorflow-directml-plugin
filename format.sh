#!/bin/bash

for f in $(find . -path ./build -prune -o -path ./third_party -prune -o -name '*.h' -or -name '*.c' -or -name '*.cpp' -or -name '*.cc'); do 
    if [ ${f} != './build' -a ${f} != './third_party' ]; then
        echo "Formatting ${f}"
        clang-format -i --style=file ${f}
    fi
done

black .
pylint build.py
pylint generate_op_defs_core.py
pylint test
pylint tfdml

echo "Done!"
