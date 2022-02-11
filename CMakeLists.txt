cmake_minimum_required(VERSION 3.20)

project(tensorflow-directml-plugin VERSION 0.0.1 LANGUAGES CXX)

# Statically link runtime library to avoid runtime dependency on Visual C++ redistributable.
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

# The C++ standard needs to be set globally to ensure Abseil targets are compiled
# with the same standard as the dependent targets.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D_SCL_SECURE_NO_WARNINGS)
add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

# PIC needs to be enabled globally for Abseil targets.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable folder organization in IDEs.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT MSVC)
    # Various protobuf files have warnings like the following:
    # "overrides a member function but is not marked 'override' [-Winconsistent-missing-override]"
    # "function declared 'noreturn' should not return [-Winvalid-noreturn]"
    add_compile_options(-Wno-inconsistent-missing-override)
    add_compile_options(-Wno-invalid-noreturn)
endif()

# Import dependencies and their targets.
include(cmake/dependencies.cmake)

# Build options/definitions that are useful to most targets. These are attached to a
# target to prevent polluting all possible targets with these settings.
add_library(common_build_props INTERFACE)
target_include_directories(common_build_props INTERFACE ${CMAKE_SOURCE_DIR})

if(WIN32)
    target_compile_definitions(common_build_props INTERFACE DML_BUILD_WINDOWS)
    target_compile_definitions(common_build_props INTERFACE WIN32_LEAN_AND_MEAN)
    target_compile_definitions(common_build_props INTERFACE NOGDI)
    target_compile_definitions(common_build_props INTERFACE NOMINMAX)
endif()

if(MSVC)
    target_compile_options(
        common_build_props 
        INTERFACE 
        /WX 
        /wd4190 # 'TF_NewWhile' has C-linkage specified, but returns UDT 'TF_WhileParams' which is incompatible with C
        /wd4005 # The version of protobuf that TensorFlow uses doesn't do an ifndef before defining WIN32_LEAN_AND_MEAN
        /Zc:preprocessor
        $<$<CONFIG:Debug>:/FS>
    )
    target_link_options(
        common_build_props
        INTERFACE
        $<$<CONFIG:Debug>:/DEBUG:FASTLINK>
    )
else()
    target_compile_options(
        common_build_props 
        INTERFACE 
        -Werror 
        -Wno-implicit-int-float-conversion
        -Wno-macro-redefined
    )
endif()

option(TFDML_TELEMETRY "Build with telemetry enabled" OFF)
if(TFDML_TELEMETRY)
    target_compile_definitions(common_build_props INTERFACE DIRECTML_ENABLE_TELEMETRY)
endif()

set(TFDML_WHEEL_VERSION_SUFFIX "" CACHE STRING "Optional text to append to the Python wheel version")

# Target comprising source that (mostly) originates from the core TF runtime and is needed
# to implement the core TFDML plugin and its kernels. There is no clear interface between
# the TF runtime and device backend/kernels in TF1.15, and many of the TFDML classes/kernels
# are ported from TF1.15 fork.
#
# WARNING: Some of the files in this library are under Apache license and largely unmodified
# from their original form. Other files simply retain the original struct/class names but are
# total rewrites. Pay careful attention to license banners when modifying code in this
# target. We may choose to further refactor these files over time to simplify layering.
add_library(
    runtime_adapter
    STATIC
    tfdml/runtime_adapter/allocator.cc
    tfdml/runtime_adapter/allocator_retry.cc
    tfdml/runtime_adapter/bcast.cc
    tfdml/runtime_adapter/bfc_allocator.cc
    tfdml/runtime_adapter/determinism.cc
    tfdml/runtime_adapter/device.cc
    tfdml/runtime_adapter/env.cc
    tfdml/runtime_adapter/env_var.cc
    tfdml/runtime_adapter/fused_eigen_output_kernels.cc
    tfdml/runtime_adapter/guarded_philox_random.cc
    tfdml/runtime_adapter/kernel_shape_util.cc
    tfdml/runtime_adapter/mirror_pad_mode.cc
    tfdml/runtime_adapter/numbers.cc
    tfdml/runtime_adapter/op_kernel_construction.cc
    tfdml/runtime_adapter/op_kernel_context.cc
    tfdml/runtime_adapter/padding.cc
    tfdml/runtime_adapter/stateless_random_ops.cc
    tfdml/runtime_adapter/status.cc
    tfdml/runtime_adapter/tensor.cc
    tfdml/runtime_adapter/tensor_format.cc
    tfdml/runtime_adapter/tensor_shape.cc
    tfdml/runtime_adapter/tensor_shape_utils.cc
    tfdml/runtime_adapter/types.cc
    tfdml/runtime_adapter/variable_lock.cc
    tfdml/runtime_adapter/xplane_builder.cc
)
target_link_libraries(
    runtime_adapter 
    PUBLIC
    common_build_props
    PRIVATE
    tensorflow_whl::lib
    INTERFACE
    absl::base
    absl::config
    absl::core_headers
    absl::flat_hash_map
    absl::flat_hash_set
    absl::hash_function_defaults
    absl::inlined_vector
    absl::node_hash_map
    absl::raw_hash_map
    absl::raw_hash_set
    absl::strings
    absl::str_format
    absl::synchronization
    absl::time
    absl::optional
    absl::span
    absl::variant
)

# Target comprising core TFDML plugin code.
add_library(
    core
    STATIC
    tfdml/core/dml_adapter.cc
    tfdml/core/dml_adapter_impl.cc
    tfdml/core/dml_bfc_allocator.cc
    tfdml/core/dml_buffer.cc
    tfdml/core/dml_buffer_region.cc
    tfdml/core/dml_command_list.cc
    tfdml/core/dml_command_queue.cc
    tfdml/core/dml_descriptor_bfc_allocator.cc
    tfdml/core/dml_descriptor_heap_allocator.cc
    tfdml/core/dml_descriptor_pool.cc
    tfdml/core/dml_device.cc
    tfdml/core/dml_device_cache.cc
    tfdml/core/dml_device_context.cc
    tfdml/core/dml_device_state.cc
    tfdml/core/dml_dso_loader.cc
    tfdml/core/dml_error_handling.cc
    tfdml/core/dml_event_queue.cc
    tfdml/core/dml_execution_context.cc
    tfdml/core/dml_guids.cc
    tfdml/core/dml_heap_allocator.cc
    tfdml/core/dml_kernel_context.cc
    tfdml/core/dml_kernel_key.cc
    tfdml/core/dml_kernel_manager.cc
    tfdml/core/dml_kernel_wrapper.cc
    tfdml/core/dml_operator_helper.cc
    tfdml/core/dml_ops_common.cc
    tfdml/core/dml_pooled_heap.cc
    tfdml/core/dml_readback_heap.cc
    tfdml/core/dml_tensor_desc.cc
    tfdml/core/dml_tracing.cc
    tfdml/core/dml_upload_heap.cc
    tfdml/core/dml_util.cc
)
target_link_libraries(
    core 
    PUBLIC 
    common_build_props
    Microsoft::DirectX-Headers
    tensorflow_whl::lib
    directml::headers
    pix_event_runtime::headers
)

# Target comprising all TFDML kernel implementations.
add_library(
    kernels
    STATIC
    tfdml/kernels/dml_addn_op.cc
    tfdml/kernels/dml_assign_variable_op.cc
    tfdml/kernels/dml_batch_norm_ops.cc
    tfdml/kernels/dml_batch_to_space_op.cc
    tfdml/kernels/dml_bias_add_op.cc
    tfdml/kernels/dml_broadcast_to_op.cc
    tfdml/kernels/dml_cast_op.cc
    tfdml/kernels/dml_concat_op.cc
    tfdml/kernels/dml_conv_ops.cc
    tfdml/kernels/dml_cwise_ops.cc
    tfdml/kernels/dml_data_format_dim_map.cc
    tfdml/kernels/dml_data_format_vec_permute.cc
    tfdml/kernels/dml_diag_op.cc
    tfdml/kernels/dml_diag_part_op.cc
    tfdml/kernels/dml_dynamic_stitch_op.cc
    tfdml/kernels/dml_extract_image_patches_op.cc
    tfdml/kernels/dml_extract_patches_helpers.cc
    tfdml/kernels/dml_extract_volume_patches_op.cc
    tfdml/kernels/dml_fill_op.cc
    tfdml/kernels/dml_gather_nd_op.cc
    tfdml/kernels/dml_gather_op.cc
    tfdml/kernels/dml_l2loss_op.cc
    tfdml/kernels/dml_lrn_ops.cc
    tfdml/kernels/dml_matmul_op.cc
    tfdml/kernels/dml_matrix_band_part_ops.cc
    tfdml/kernels/dml_mirror_pad_grad_op.cc
    tfdml/kernels/dml_pack_op.cc
    tfdml/kernels/dml_pad_op.cc
    tfdml/kernels/dml_pooling_ops.cc
    tfdml/kernels/dml_random_ops.cc
    tfdml/kernels/dml_range_op.cc
    tfdml/kernels/dml_reduce_ops.cc
    tfdml/kernels/dml_relu_ops.cc
    tfdml/kernels/dml_resize_op.cc
    tfdml/kernels/dml_scatter_nd_ops.cc
    tfdml/kernels/dml_scatter_ops.cc
    tfdml/kernels/dml_select_op.cc
    tfdml/kernels/dml_slice_op.cc
    tfdml/kernels/dml_space_depth_ops.cc
    tfdml/kernels/dml_space_to_batch_op.cc
    tfdml/kernels/dml_sparse_xent_op.cc
    tfdml/kernels/dml_strided_slice_op.cc
    tfdml/kernels/dml_tile_op.cc
    tfdml/kernels/dml_topk_op.cc
    tfdml/kernels/dml_training_ops.cc
    tfdml/kernels/dml_transpose_op.cc
    tfdml/kernels/dml_xent_op.cc
    tfdml/kernels/dml_zeros_like_op.cc
)
target_link_libraries(
    kernels
    PUBLIC
    runtime_adapter
    core
)
target_precompile_headers(
    kernels 
    PRIVATE 
    tfdml/kernels/pch.h
)

# Target to build the TF plugin module.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$\{ORIGIN\}/directml")
add_library(
    tfdml_plugin
    SHARED
    tfdml/plugin/plugin_device.cc
    tfdml/plugin/plugin_kernel.cc
    tfdml/plugin/plugin_profiler.cc
)
target_link_libraries(
    tfdml_plugin 
    PRIVATE
    common_build_props
    core
    kernels
    runtime_adapter
    protobuf::libprotobuf
)
set_target_properties(tfdml_plugin PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)

# The CMake project name and version determines the python wheel version. Save these
# values to files so the setup.py script can read them.
set(pkg_py "cp${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
string(REPLACE "-" "_" pkg_name "${CMAKE_PROJECT_NAME}")
set(pkg_version "${CMAKE_PROJECT_VERSION}${TFDML_WHEEL_VERSION_SUFFIX}")
if(WIN32)
    set(pkg_platform "win_amd64")
else()
    set(pkg_platform "linux_x86_64")
endif()
set(pkg_full_name "${pkg_name}-${pkg_version}-${pkg_py}-${pkg_py}-${pkg_platform}.whl")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/TFDML_WHEEL_NAME ${CMAKE_PROJECT_NAME})
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/TFDML_WHEEL_VERSION ${pkg_version})

add_custom_command(
    OUTPUT 
        ${pkg_full_name}
    COMMAND 
        ${Python_EXECUTABLE} 
        ${CMAKE_SOURCE_DIR}/tfdml/wheel/build_wheel.py
        --plugin_path $<TARGET_FILE:tfdml_plugin>
        --build_dir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        tfdml_plugin
        tfdml/wheel/build_wheel.py
        tfdml/wheel/setup.py
        tfdml/wheel/README
        tfdml/wheel/MANIFEST.in
    WORKING_DIRECTORY
        ${CMAKE_SOURCE_DIR}
    COMMENT
        "Creating ${pkg_full_name}"
)
add_custom_target(tfdml_plugin_wheel ALL DEPENDS ${pkg_full_name})