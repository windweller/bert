#!/usr/bin/env bash

# http://linuxcommand.org/lc3_wss0120.php

if [ $# -eq 0 ]
  then
    echo "Need to supply argument: -t pdtb_im | pdtb_imex"
    exit 1
fi

while [ "$1" != "" ]; do
    case $1 in
        -t | --task )           shift
                                taskname=$1
                                ;;
        * )                     echo "not a valid option"
                                exit 1
    esac
    shift
done

echo "here!"
echo "here!!"