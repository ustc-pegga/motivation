import os
import sys
os.system("ndk-build")
os.system("adb push ../jni/lib /data/local/tmp")
os.system("adb push ../libs/arm64-v8a/mindspore_op /data/local/tmp")
