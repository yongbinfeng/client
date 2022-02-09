// Targeted by JavaCPP version 1.5.7-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.tritonserver.tritonserver;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;
import org.bytedeco.cuda.cublas.*;
import static org.bytedeco.cuda.global.cublas.*;
import org.bytedeco.cuda.cudnn.*;
import static org.bytedeco.cuda.global.cudnn.*;
import org.bytedeco.cuda.nvrtc.*;
import static org.bytedeco.cuda.global.nvrtc.*;
import org.bytedeco.tensorrt.nvinfer.*;
import static org.bytedeco.tensorrt.global.nvinfer.*;
import org.bytedeco.tensorrt.nvinfer_plugin.*;
import static org.bytedeco.tensorrt.global.nvinfer_plugin.*;
import org.bytedeco.tensorrt.nvonnxparser.*;
import static org.bytedeco.tensorrt.global.nvonnxparser.*;
import org.bytedeco.tensorrt.nvparsers.*;
import static org.bytedeco.tensorrt.global.nvparsers.*;

import static org.bytedeco.tritonserver.global.tritonserver.*;

// #endif
// #endif

@Opaque @Properties(inherit = org.bytedeco.tritonserver.presets.tritonserver.class)
public class TRITONBACKEND_MemoryManager extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TRITONBACKEND_MemoryManager() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TRITONBACKEND_MemoryManager(Pointer p) { super(p); }
}
