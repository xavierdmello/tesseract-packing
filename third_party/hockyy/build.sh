#!/bin/zsh
# Build hockyy's polish/polishx/pack into third_party/hockyy/bin (Xcode toolchain + SDK).
cd "$(dirname "$0")"; mkdir -p bin
CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
for f in polish polishx pack; do $CXX -isysroot $SDK -Icompat -std=c++17 -O3 -ffast-math -mcpu=native -pthread $f.cpp -o bin/$f || exit 1; done
echo built: bin/*
