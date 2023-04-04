cmake_minimum_required(VERSION 3.20)

project(tensorflow-directml-plugin VERSION 0.5.0 LANGUAGES CXX)

# Statically link runtime library to avoid runtime dependency on Visual C++ redistributable.
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

# The C++ standard needs to be set globally to ensure Abseil targets are compiled
# with the same standard as the dependent targets.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D_SCL_SECURE_NO_WARNINGS)
add_definitions(-D_SILENCE_ALL_CXX14_DEPRECATION_WARNINGS)

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
    add_compile_options(-Wno-unknown-warning-option)
endif()

cmake_policy(SET CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

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
        /wd5105 # macro expansion producing 'defined' has undefined behavior in winbase.h
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
option(TFDML_TELEMETRY_PROVIDER_GROUP_GUID "The GUID of the telemetry provider group to use in the format '00000000-0000-0000-0000-000000000000'" "")
if(TFDML_TELEMETRY)
    target_compile_definitions(common_build_props INTERFACE DIRECTML_ENABLE_TELEMETRY)

    if (TFDML_TELEMETRY_PROVIDER_GROUP_GUID)
        # Convert the GUID representation to a comma separated list suitable for the TraceLoggingOptionGroup define
        string(REPLACE "-" "" TFDML_TELEMETRY_PROVIDER_GROUP_GUID ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID})

        string(LENGTH ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} TFDML_TELEMETRY_PROVIDER_GROUP_GUID_LENGTH)

        if (NOT TFDML_TELEMETRY_PROVIDER_GROUP_GUID_LENGTH EQUAL 32)
            message(FATAL_ERROR "TFDML_TELEMETRY_PROVIDER_GROUP_GUID needs to have 32 alphanumerical characters, but ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_LENGTH} were found.")
        endif()

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 0 8 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART1)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART1 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 8 4 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART2)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART2 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 12 4 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART3)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART3 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 16 2 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART4)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART4 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 18 2 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART5)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART5 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 20 2 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART6)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART6 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 22 2 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART7)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART7 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 24 2 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART8)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART8 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 26 2 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART9)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART9 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 28 2 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART10)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART10 "0x")

        string(SUBSTRING ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID} 30 2 TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART11)
        string(PREPEND TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART11 "0x")

        string(
            JOIN
            ", "
            TFDML_TELEMETRY_PROVIDER_GROUP_GUID
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART1}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART2}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART3}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART4}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART5}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART6}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART7}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART8}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART9}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART10}
            ${TFDML_TELEMETRY_PROVIDER_GROUP_GUID_PART11}
        )

        target_compile_definitions(common_build_props INTERFACE DIRECTML_TELEMETRY_PROVIDER_GROUP_GUID=${TFDML_TELEMETRY_PROVIDER_GROUP_GUID})
    else()
        message(FATAL_ERROR "When TFDML_TELEMETRY is enabled, TFDML_TELEMETRY_PROVIDER_GROUP_GUID needs to be set.")
    endif()
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
    tfdml/runtime_adapter/op_defs_core.cc
    tfdml/runtime_adapter/op_kernel_construction.cc
    tfdml/runtime_adapter/op_kernel_context.cc
    tfdml/runtime_adapter/padding.cc
    tfdml/runtime_adapter/path.cc
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
    PRIVATE
    tensorflow_protos
    common_build_props
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
target_include_directories(
    runtime_adapter
    PRIVATE
    ${tensorflow_whl_SOURCE_DIR}/tensorflow/include
    ${abseil_SOURCE_DIR}
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
    tfdml/core/dml_device_manager.cc
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
    tfdml/core/dml_tagged_pointer.cc
    tfdml/core/dml_tensor_desc.cc
    tfdml/core/dml_tracing.cc
    tfdml/core/dml_upload_heap.cc
    tfdml/core/dml_util.cc
)
target_include_directories(
    core
    PRIVATE
    ${abseil_SOURCE_DIR}
    ${tensorflow_whl_SOURCE_DIR}/tensorflow/include
)
target_link_libraries(
    core 
    PRIVATE
    tensorflow_protos
    common_build_props
    Microsoft::DirectX-Headers
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
    tfdml/kernels/dml_bitcast_op.cc
    tfdml/kernels/dml_broadcast_to_op.cc
    tfdml/kernels/dml_cast_op.cc
    tfdml/kernels/dml_check_numerics_op.cc
    tfdml/kernels/dml_concat_op.cc
    tfdml/kernels/dml_conv_ops.cc
    tfdml/kernels/dml_crop_and_resize_grad_boxes_op.cc
    tfdml/kernels/dml_crop_and_resize_grad_image_op.cc
    tfdml/kernels/dml_crop_and_resize_op.cc
    tfdml/kernels/dml_cross_op.cc
    tfdml/kernels/dml_cudnn_rnn_ops.cc
    tfdml/kernels/dml_cwise_ops.cc
    tfdml/kernels/dml_data_format_dim_map.cc
    tfdml/kernels/dml_data_format_vec_permute.cc
    tfdml/kernels/dml_deepcopy_op.cc
    tfdml/kernels/dml_diag_op.cc
    tfdml/kernels/dml_diag_part_op.cc
    tfdml/kernels/dml_dynamic_stitch_op.cc
    tfdml/kernels/dml_empty_op.cc
    tfdml/kernels/dml_extract_image_patches_op.cc
    tfdml/kernels/dml_extract_patches_helpers.cc
    tfdml/kernels/dml_extract_volume_patches_op.cc
    tfdml/kernels/dml_fill_op.cc
    tfdml/kernels/dml_gather_nd_op.cc
    tfdml/kernels/dml_gather_op.cc
    tfdml/kernels/dml_gru_ops.cc
    tfdml/kernels/dml_image_ops.cc
    tfdml/kernels/dml_in_topk_op.cc
    tfdml/kernels/dml_inplace_ops.cc
    tfdml/kernels/dml_l2loss_op.cc
    tfdml/kernels/dml_lrn_ops.cc
    tfdml/kernels/dml_lstm_ops.cc
    tfdml/kernels/dml_matmul_op.cc
    tfdml/kernels/dml_matrix_band_part_ops.cc
    tfdml/kernels/dml_matrix_diag_helpers.cc
    tfdml/kernels/dml_matrix_diag_ops.cc
    tfdml/kernels/dml_matrix_diag_part_ops.cc
    tfdml/kernels/dml_matrix_set_diag_ops.cc
    tfdml/kernels/dml_mirror_pad_grad_op.cc
    tfdml/kernels/dml_one_hot_op.cc
    tfdml/kernels/dml_ones_like_op.cc
    tfdml/kernels/dml_pack_op.cc
    tfdml/kernels/dml_pad_op.cc
    tfdml/kernels/dml_parallel_concat_ops.cc
    tfdml/kernels/dml_pooling_ops.cc
    tfdml/kernels/dml_random_ops.cc
    tfdml/kernels/dml_range_op.cc
    tfdml/kernels/dml_reduce_ops.cc
    tfdml/kernels/dml_relu_ops.cc
    tfdml/kernels/dml_resize_grad_ops.cc
    tfdml/kernels/dml_resize_op.cc
    tfdml/kernels/dml_reverse_op.cc
    tfdml/kernels/dml_reverse_sequence_op.cc
    tfdml/kernels/dml_roll_op.cc
    tfdml/kernels/dml_scan_ops.cc
    tfdml/kernels/dml_scatter_nd_ops.cc
    tfdml/kernels/dml_scatter_ops.cc
    tfdml/kernels/dml_segment_reduction_ops.cc
    tfdml/kernels/dml_select_op.cc
    tfdml/kernels/dml_slice_op.cc
    tfdml/kernels/dml_snapshot_op.cc
    tfdml/kernels/dml_space_depth_ops.cc
    tfdml/kernels/dml_space_to_batch_op.cc
    tfdml/kernels/dml_sparse_xent_op.cc
    tfdml/kernels/dml_split_op.cc
    tfdml/kernels/dml_strided_slice_op.cc
    tfdml/kernels/dml_swapping_ops.cc
    tfdml/kernels/dml_tile_op.cc
    tfdml/kernels/dml_topk_op.cc
    tfdml/kernels/dml_training_ops.cc
    tfdml/kernels/dml_transpose_op.cc
    tfdml/kernels/dml_unpack_op.cc
    tfdml/kernels/dml_where_op.cc
    tfdml/kernels/dml_xent_op.cc
    tfdml/kernels/dml_zeros_like_op.cc
)
target_include_directories(
    kernels
    PRIVATE
    ${tensorflow_whl_SOURCE_DIR}/tensorflow/include
    ${abseil_SOURCE_DIR}
)
target_link_libraries(
    kernels
    PRIVATE
    common_build_props
    runtime_adapter
    core
    tensorflow_protos
    Microsoft::DirectX-Headers
    directml_redist::headers
    directmlx::headers
)
target_precompile_headers(
    kernels 
    PRIVATE 
    tfdml/kernels/pch.h
)

# Target comprising all TFDML grappler optimizations
add_library(
    optimizer
    STATIC
    tfdml/optimizer/device_name_utils.cc
    tfdml/optimizer/device_type.cc
    tfdml/optimizer/graph_optimizer.cc
    tfdml/optimizer/graph_properties.cc
    tfdml/optimizer/graph_view.cc
    tfdml/optimizer/grappler_item.cc
    tfdml/optimizer/hash.cc
    tfdml/optimizer/op_registry.cc
    tfdml/optimizer/op_types.cc
    tfdml/optimizer/optimizer_runner.cc
    tfdml/optimizer/perm_utils.cc
    tfdml/optimizer/proto_buffer_helpers.cc
    tfdml/optimizer/remapper.cc
    tfdml/optimizer/tensor_id.cc
    tfdml/optimizer/tensor_proto_util.cc
    tfdml/optimizer/transpose_remover.cc
    tfdml/optimizer/utils.cc
)
target_include_directories(
    optimizer
    PRIVATE
    ${tensorflow_whl_SOURCE_DIR}/tensorflow/include
    ${abseil_SOURCE_DIR}
)
target_link_libraries(
    optimizer
    PRIVATE
    common_build_props
    runtime_adapter
    core
    tensorflow_protos
)

# Target to build the TF plugin module.
add_library(
    tfdml_plugin
    SHARED
    tfdml/plugin/plugin_device.cc
    tfdml/plugin/plugin_kernel.cc
    tfdml/plugin/plugin_optimizer.cc
    tfdml/plugin/plugin_profiler.cc
    tfdml.def
)
target_include_directories(
    tfdml_plugin
    PRIVATE
    ${tensorflow_whl_SOURCE_DIR}/tensorflow/include
    ${abseil_SOURCE_DIR}
)
target_link_libraries(
    tfdml_plugin 
    PRIVATE
    common_build_props
    core
    kernels
    optimizer
    runtime_adapter
    tensorflow_protos
    Microsoft::DirectX-Headers
    directml::headers
    tensorflow_python_libs
)
set_target_properties(
    tfdml_plugin
    PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$\{ORIGIN\}/directml"
)

add_library(
    tfdml_plugin_framework
    SHARED
    tfdml/plugin/plugin_device.cc
    tfdml/plugin/plugin_kernel.cc
    tfdml/plugin/plugin_optimizer.cc
    tfdml/plugin/plugin_profiler.cc
    tfdml.def
)
target_include_directories(
    tfdml_plugin_framework
    PRIVATE
    ${tensorflow_whl_SOURCE_DIR}/tensorflow/include
    ${abseil_SOURCE_DIR}
)
target_link_libraries(
    tfdml_plugin_framework 
    PRIVATE
    common_build_props
    core
    kernels
    optimizer
    runtime_adapter
    tensorflow_protos
    Microsoft::DirectX-Headers
    directml::headers
    tensorflow_framework_libs
    libprotobuf
)
set_target_properties(
    tfdml_plugin_framework
    PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$\{ORIGIN\}/directml"
)

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

enable_testing()

add_executable(
    c_api_tests
    test/c/c_api_tests.cc
)
target_link_libraries(
    c_api_tests
    PRIVATE
    tfdml_plugin_framework
    tensorflow_framework_libs
    GTest::gtest_main
    ${CMAKE_DL_LIBS}
)
target_include_directories(
    c_api_tests
    PRIVATE
    ${tensorflow_whl_SOURCE_DIR}/tensorflow/include
    ${abseil_SOURCE_DIR}
)
set_target_properties(
    c_api_tests
    PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$\{ORIGIN\}"
)

add_custom_command(
    OUTPUT 
        ${pkg_full_name}
    COMMAND 
        ${Python_EXECUTABLE} 
        ${CMAKE_SOURCE_DIR}/tfdml/wheel/build_wheel.py
        --plugin_paths
            $<TARGET_FILE:tfdml_plugin>
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

# TODO: Make available on Windows once the TensorFlow C API exports all the necessary symbols
# TF #40927951
# if(NOT WIN32)
# Get the DirectML SHA
file(READ "${directml_redist_SOURCE_DIR}/include/DirectMLConfig.h" DIRECTML_CONFIG)
string(REGEX MATCH "#define DIRECTML_SOURCE_VERSION \"([abcdef0-9]+)\"" REGEX_MATCH ${DIRECTML_CONFIG})
set(DIRECTML_SHA ${CMAKE_MATCH_1})

if(WIN32)
    set(framework_full_name "${CMAKE_PROJECT_NAME}-${pkg_platform}-${pkg_version}.zip")
else()
    set(framework_full_name "lib${CMAKE_PROJECT_NAME}-${pkg_platform}-${pkg_version}.tar.gz")
endif()

add_custom_command(
    OUTPUT 
        ${framework_full_name}
    COMMAND
        ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tfdml_plugin_framework> .
    COMMAND
        ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/LICENSE .
    COMMAND
        ${CMAKE_COMMAND} -E make_directory ./directml
    COMMAND
        ${CMAKE_COMMAND} -E copy
        $<$<BOOL:${UNIX}>:${directml_redist_SOURCE_DIR}/bin/x64-linux/libdirectml.so>
        $<$<BOOL:${UNIX}>:./directml/libdirectml.${DIRECTML_SHA}.so>
        $<$<BOOL:${WIN32}>:${directml_redist_SOURCE_DIR}/bin/x64-win/DirectML.dll>
        $<$<BOOL:${WIN32}>:./directml/DirectML.${DIRECTML_SHA}.dll>
    COMMAND
        ${CMAKE_COMMAND} -E copy ${directml_redist_SOURCE_DIR}/LICENSE.txt ./directml/DirectML_LICENSE.txt
    COMMAND
        ${CMAKE_COMMAND} -E copy ${directml_redist_SOURCE_DIR}/ThirdPartyNotices.txt ./directml/DirectML_ThirdPartyNotices.txt
    COMMAND
        $<$<BOOL:${WIN32}>:${CMAKE_COMMAND}> $<$<BOOL:${WIN32}>:-E> $<$<BOOL:${WIN32}>:copy>
        $<$<BOOL:${WIN32}>:${pix_event_runtime_SOURCE_DIR}/bin/x64/WinPixEventRuntime.dll>
        $<$<BOOL:${WIN32}>:./directml>
    COMMAND
        $<$<BOOL:${WIN32}>:${CMAKE_COMMAND}> $<$<BOOL:${WIN32}>:-E> $<$<BOOL:${WIN32}>:copy>
        $<$<BOOL:${WIN32}>:${pix_event_runtime_SOURCE_DIR}/license.txt>
        $<$<BOOL:${WIN32}>:./directml/WinPixEventRuntime_LICENSE.txt>
    COMMAND
        $<$<BOOL:${WIN32}>:${CMAKE_COMMAND}> $<$<BOOL:${WIN32}>:-E> $<$<BOOL:${WIN32}>:copy>
        $<$<BOOL:${WIN32}>:${pix_event_runtime_SOURCE_DIR}/ThirdPartyNotices.txt>
        $<$<BOOL:${WIN32}>:./directml/WinPixEventRuntime_ThirdPartyNotices.txt>
    COMMAND
        ${CMAKE_COMMAND} -E tar $<$<BOOL:${WIN32}>:"cfv"> $<$<BOOL:${UNIX}>:"cfvz"> "${framework_full_name}" $<$<BOOL:${WIN32}>:--format=zip> --
        $<TARGET_FILE_NAME:tfdml_plugin_framework>
        LICENSE
        directml/DirectML_LICENSE.txt
        directml/DirectML_ThirdPartyNotices.txt
        $<$<BOOL:${UNIX}>:directml/libdirectml.${DIRECTML_SHA}.so>
        $<$<BOOL:${WIN32}>:directml/DirectML.${DIRECTML_SHA}.dll>
        $<$<BOOL:${WIN32}>:directml/WinPixEventRuntime.dll>
    DEPENDS
        tfdml_plugin_framework
        LICENSE
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT
        "Creating ${framework_full_name}"
)
add_custom_target(tfdml_plugin_framework_zip ALL DEPENDS ${framework_full_name})

set(c_api_tests_full_name "${pkg_name}-${pkg_version}-${pkg_platform}-c-api-tests.zip")
add_custom_command(
    OUTPUT 
        ${c_api_tests_full_name}
    COMMAND
        ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:tfdml_plugin_framework>
        $<TARGET_FILE:c_api_tests>
        $<$<BOOL:${UNIX}>:${tensorflow_framework_SOURCE_DIR}/lib/libtensorflow.so.2>
        $<$<BOOL:${UNIX}>:${tensorflow_framework_SOURCE_DIR}/lib/libtensorflow_framework.so.2>
        $<$<BOOL:${WIN32}>:${tensorflow_framework_SOURCE_DIR}/lib/tensorflow.dll>
        .
    COMMAND
        ${CMAKE_COMMAND} -E make_directory ./directml
    COMMAND
        ${CMAKE_COMMAND} -E copy
        $<$<BOOL:${UNIX}>:${directml_redist_SOURCE_DIR}/bin/x64-linux/libdirectml.so>
        $<$<BOOL:${UNIX}>:./directml/libdirectml.${DIRECTML_SHA}.so>
        $<$<BOOL:${WIN32}>:${directml_redist_SOURCE_DIR}/bin/x64-win/DirectML.dll>
        $<$<BOOL:${WIN32}>:./directml/DirectML.${DIRECTML_SHA}.dll>
    COMMAND
        $<$<BOOL:${WIN32}>:${CMAKE_COMMAND}> $<$<BOOL:${WIN32}>:-E> $<$<BOOL:${WIN32}>:copy>
        $<$<BOOL:${WIN32}>:${pix_event_runtime_SOURCE_DIR}/bin/x64/WinPixEventRuntime.dll>
        $<$<BOOL:${WIN32}>:./directml>
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${squeezenet_model_SOURCE_DIR} ./squeezenet_model
    COMMAND
        ${CMAKE_COMMAND} -E tar "cfv" "${c_api_tests_full_name}" --format=zip --
        $<TARGET_FILE_NAME:tfdml_plugin_framework>
        $<TARGET_FILE_NAME:c_api_tests>
        $<$<BOOL:${UNIX}>:libtensorflow.so.2>
        $<$<BOOL:${UNIX}>:libtensorflow_framework.so.2>
        $<$<BOOL:${UNIX}>:directml/libdirectml.${DIRECTML_SHA}.so>
        $<$<BOOL:${WIN32}>:tensorflow.dll>
        $<$<BOOL:${WIN32}>:directml/DirectML.${DIRECTML_SHA}.dll>
        $<$<BOOL:${WIN32}>:directml/WinPixEventRuntime.dll>
        squeezenet_model/squeezenet.pb
    DEPENDS
        c_api_tests
        tfdml_plugin_framework
        tensorflow_framework_libs
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT
        "Creating ${c_api_tests_full_name}"
)
add_custom_target(c_api_tests_zip ALL DEPENDS ${c_api_tests_full_name})
# endif()
