cmake_minimum_required(VERSION 2.6)
project(ltrckr)

file(GLOB ltrckr_src src/ltrckr/*.cpp)
#file(GLOB sms_src src/sms/*.c)
#file(GLOB sndobj_src src/sndobj/*.cpp)
file(GLOB loris_src src/loris/*.C)
# file(GLOB mq_src src/mq/*.cpp)
list(APPEND source_files ${ltrckr_src}
                         #${sndobj_src}
                         #${sms_src}
                         ${loris_src}
                         #${mq_src})

file(GLOB ltrckr_include src/ltrckr/*.h)
#set(sms_include src/sms/sms.h)
#file(GLOB sndobj_include src/sndobj/*.h)
file(GLOB loris_include src/loris/*.h)
#file(GLOB mq_include src/mq/*.h)
list(APPEND include_files ${ltrckr_include}
                          #${sms_include}
                          #${sndobj_include}
                          ${loris_include}
                          #${mq_include})

add_definitions(-DHAVE_FFTW3_H)
set(libs m fftw3 gsl gslcblas)

include_directories(src/ltrckr src/loris)

add_library(ltrckr SHARED ${source_files})
target_link_libraries(ltrckr ${libs})

install(TARGETS simpl LIBRARY DESTINATION lib)
install(FILES ${include_files} DESTINATION include/ltrckr)