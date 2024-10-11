cmake_minimum_required(VERSION 3.22)
function(add_custom_file TARGET SRC_PATH DST_REL_PATH)
    # find_program(TEXTPROG texture_processor)
    # sanitize input paths
    file(REAL_PATH ${SRC_PATH} current-file-source-path EXPAND_TILDE)
    set(current-destination-path ${CMAKE_CURRENT_BINARY_DIR}/${DST_REL_PATH})

    # Add a custom command for possible processing.
    get_filename_component(current-output-dir ${current-destination-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})
    add_custom_command(
            OUTPUT ${current-destination-path}
            #COMMAND ${TEXTPROG} -args ${current-file-source-path} ${current-file-source-path}
            COMMAND ${CMAKE_COMMAND} -E copy ${current-file-source-path} ${current-destination-path}
            DEPENDS ${current-file-source-path}
            IMPLICIT_DEPENDS CXX ${current-file-source-path}
            VERBATIM)

    # Make sure our build depends on this output.
    set_source_files_properties(${current-destination-path} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${current-destination-path})
endfunction(add_custom_file)