cmake_minimum_required(VERSION 3.14)
project(BitWizzard VERSION "0.0.0")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

###
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SaneCompilerOptions)

###
option(BW_PLUGIN_COPY "Copy plugin after build" ON)
option(BW_PLUGIN_LTO "Enable plugin LTO" OFF)
option(BW_PLUGIN_FORCE_DEBUG "Force plugin to use debug mode" OFF)

###
include(FetchContent)

FetchContent_Declare(juce
    GIT_REPOSITORY "https://github.com/juce-framework/JUCE.git"
    GIT_TAG "e27194272de76707cddb4bde14638bc3bd5793e5")

FetchContent_MakeAvailable(juce)

###
add_library(WDL STATIC EXCLUDE_FROM_ALL
  "plugin/WDLex/resampleMOD.cpp"
  "plugin/WDLex/resampleMOD.h")
target_compile_definitions(WDL PUBLIC
  "WDL_RESAMPLE_TYPE=float"
  "WDL_RESAMPLE_RATIONAL")
target_include_directories(WDL PUBLIC
  "plugin"
  "thirdparty/WDL")

###
juce_add_plugin(bitwiz_plugin
  PLUGIN_CODE "BtWz"
  PLUGIN_MANUFACTURER_CODE "BtWz"
  PRODUCT_NAME "BitWizzard"
  COMPANY_NAME "Unseen DSP"
  FORMATS VST3 AU
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  NEEDS_CURL FALSE
  NEEDS_WEB_BROWSER FALSE
  VST3_CATEGORIES "Fx"
  AU_MAIN_TYPE "kAudioUnitType_Effect"
  COPY_PLUGIN_AFTER_BUILD "${BW_PLUGIN_COPY}")

target_sources(bitwiz_plugin
  PRIVATE
    "plugin/definitions.hpp"
    "plugin/processor.cpp"
    "plugin/processor.hpp")

target_compile_definitions(bitwiz_plugin
  PUBLIC
    "JUCE_WEB_BROWSER=0"
    "JUCE_USE_CURL=0"
    "JUCE_VST3_CAN_REPLACE_VST2=0"
    "JUCE_DISPLAY_SPLASH_SCREEN=0")

target_link_libraries(bitwiz_plugin
  PRIVATE
      WDL
      juce::juce_audio_processors
      juce::juce_gui_basics
      juce::juce_gui_extra
      juce::juce_opengl
      juce::juce_recommended_config_flags
      juce::juce_recommended_warning_flags)

if(BW_PLUGIN_LTO)
  target_link_libraries(bitwiz_plugin PRIVATE juce::juce_recommended_lto_flags)
endif()

if(BW_PLUGIN_FORCE_DEBUG)
  target_compile_definitions(bitwiz_plugin PRIVATE "JUCE_FORCE_DEBUG=1")
endif()
