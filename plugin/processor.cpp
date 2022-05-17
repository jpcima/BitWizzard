//Copyright CC) 2022- Jean Pierre Cimalando <jp-dev@gmx.com>
//SPDX-License-Identifier: GPL-3.0-or-later

#include "processor.hpp"
#include "definitions.hpp"
#include "WDLex/resampleMOD.h"
#include <vector>
#include <memory>
#include <algorithm>

struct BWProcessor::Impl {
    BWProcessor *m_self = nullptr;
    std::unique_ptr<juce::AudioProcessorValueTreeState> m_treestate;

    // chip DSP stuff
    juce::Array<int16_t> m_last_q_sample;

    // resampling stuff
    int m_active_chip_rate = -1;
    WDL_Resampler m_downsampler;
    WDL_Resampler m_upsampler;
    juce::Array<float> m_last_dsp_outs;
    juce::Array<float> m_downsampler_out;
    juce::Array<float> m_upsampler_out;
    juce::AudioBuffer<float> m_dsp_in;
    juce::AudioBuffer<float> m_dsp_out;

    struct {
        std::atomic<float> *chip_rate = nullptr;
        std::atomic<float> *quant_bits = nullptr;
        std::atomic<float> *quant_scale = nullptr;
        std::atomic<float> *delta_speed = nullptr;
        std::atomic<float> *delta_noise = nullptr;
    } m_param;
};

BWProcessor::BWProcessor()
    : juce::AudioProcessor(BusesProperties()
                           .withInput("Input", juce::AudioChannelSet::stereo(), true)
                           .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    Impl *impl = new Impl;
    m_impl.reset(impl);

    //
    auto unitFormatterInt = [](const char *unit) -> std::function<juce::String(int, int)> {
        juce::String suffix = ' ' + juce::String{juce::CharPointer_UTF8{unit}};
        return [suffix](int value, int maxlen) -> juce::String {
            (void)maxlen;
            return juce::String(value) + suffix;
        };
    };
    auto unitFormatterInt2 = [](const char *unit, const char *plural) -> std::function<juce::String(int, int)> {
        juce::String suffix = ' ' + juce::String{juce::CharPointer_UTF8{unit}};
        juce::String suffix2 = ' ' + juce::String{juce::CharPointer_UTF8{plural}};
        return [suffix, suffix2](int value, int maxlen) -> juce::String {
            (void)maxlen;
            return juce::String(value) + ((std::abs(value) > 1) ? suffix2 : suffix);
        };
    };
    auto unitFormatterFloat = [](const char *unit) -> std::function<juce::String(float, int)> {
        juce::String suffix = ' ' + juce::String{juce::CharPointer_UTF8{unit}};
        return [suffix](float value, int maxlen) -> juce::String {
            (void)maxlen;
            return juce::String(value) + suffix;
        };
    };

    //
    impl->m_self = this;
    impl->m_treestate = std::make_unique<juce::AudioProcessorValueTreeState>(
        *this, nullptr, "PARAMETERS",
        juce::AudioProcessorValueTreeState::ParameterLayout
        {
            std::make_unique<juce::AudioParameterInt>(
                "chip-rate", "Chip rate",
                BW_min_chip_rate, BW_max_chip_rate, BW_def_chip_rate,
                juce::String{}, unitFormatterInt("Hz")),
            std::make_unique<juce::AudioParameterFloat>(
                "quant-bits", "Quantization bits",
                juce::NormalisableRange<float>{2, 12}, 7,
                juce::String{}, juce::AudioProcessorParameter::genericParameter,
                unitFormatterFloat("bits")),
            std::make_unique<juce::AudioParameterFloat>(
                "quant-scale", "Scale gain",
                juce::NormalisableRange<float>{-20, +20}, 0,
                juce::String{}, juce::AudioProcessorParameter::genericParameter,
                unitFormatterFloat("dB")),
            std::make_unique<juce::AudioParameterInt>(
                "delta-speed", "Delta speed",
                1, 16, 1,
                juce::String{}, unitFormatterInt2("step", "steps")),
            std::make_unique<juce::AudioParameterFloat>(
                "delta-noise", "Delta noise",
                juce::NormalisableRange<float>{0, 2}, 1,
                juce::String{}, juce::AudioProcessorParameter::genericParameter),
        });

    impl->m_param.chip_rate = impl->m_treestate->getRawParameterValue("chip-rate");
    impl->m_param.quant_bits = impl->m_treestate->getRawParameterValue("quant-bits");
    impl->m_param.quant_scale = impl->m_treestate->getRawParameterValue("quant-scale");
    impl->m_param.delta_speed = impl->m_treestate->getRawParameterValue("delta-speed");
    impl->m_param.delta_noise = impl->m_treestate->getRawParameterValue("delta-noise");
}

BWProcessor::~BWProcessor()
{
}

//==============================================================================
void BWProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    (void)samplesPerBlock;

    Impl *impl = m_impl.get();
    int nchan = getBusesLayout().getMainInputChannels();

    // chip DSP stuff
    impl->m_last_q_sample.clearQuick();
    impl->m_last_q_sample.resize(nchan);

    // resampling stuff
    impl->m_active_chip_rate = -1;

    int down_bufsize = 4 + BW_max_segment * BW_max_chip_rate / (int)sampleRate;
    int up_bufsize = 4 + BW_max_segment * (int)sampleRate / BW_min_chip_rate;

    impl->m_downsampler_out.resize(nchan * down_bufsize);
    impl->m_upsampler_out.resize(nchan * up_bufsize);

    impl->m_dsp_in = juce::AudioBuffer<float>{nchan, down_bufsize};
    impl->m_dsp_out = juce::AudioBuffer<float>{nchan, down_bufsize};

    impl->m_last_dsp_outs.clearQuick();
    impl->m_last_dsp_outs.resize(nchan);
}

void BWProcessor::releaseResources()
{
}

void BWProcessor::processBlock(juce::AudioBuffer<float> &buffer, juce::MidiBuffer &midiMessages)
{
    (void)midiMessages;

    int nframes = buffer.getNumSamples();
    if (nframes < 1)
        return;

    juce::ScopedNoDenormals nodenormals;

    Impl *impl = m_impl.get();

    // update the sample rate parameter if changed
    int chip_rate = (int)impl->m_param.chip_rate->load(std::memory_order_relaxed);
    chip_rate = juce::jlimit(BW_min_chip_rate, BW_max_chip_rate, chip_rate);
    if (impl->m_active_chip_rate != chip_rate) {
        int sampleRate = (int)getSampleRate();
        impl->m_downsampler.SetRates(sampleRate, chip_rate);
        impl->m_downsampler.Reset();
        impl->m_upsampler.SetRates(chip_rate, sampleRate);
        impl->m_upsampler.Reset();
        impl->m_active_chip_rate = chip_rate;
    }

    // process in segments
    int index = 0;
    while (index < nframes) {
        int bs = juce::jmin(BW_max_segment, nframes - index);
        processSegment(buffer, index, bs);
        index += bs;
    }
}

void BWProcessor::processBlock(juce::AudioBuffer<double> &buffer, juce::MidiBuffer &midiMessages)
{
    (void)buffer;
    (void)midiMessages;
    jassertfalse;
}

bool BWProcessor::supportsDoublePrecisionProcessing() const
{
    return false;
}

static inline void interleave(const float *const *inputs, float *const outputs, int nch, int nframes)
{
    for (int ch = 0; ch < nch; ++ch) {
        const float *src = inputs[ch];
        float *dst = outputs + ch;
        for (int i = 0; i < nframes; ++i)
            dst[i * nch] = src[i];
    }
}

static inline void deinterleave(const float *const inputs, float *const *outputs, int nch, int nframes)
{
    for (int ch = 0; ch < nch; ++ch) {
        const float *src = inputs + ch;
        float *dst = outputs[ch];
        for (int i = 0; i < nframes; ++i)
            dst[i] = src[i * nch];
    }
}

void BWProcessor::processSegment(juce::AudioBuffer<float> &buffer, int segstart, int segsize)
{
    Impl *impl = m_impl.get();
    int nchan = buffer.getNumChannels();
    const float **inputs = (const float **)alloca((size_t)nchan * sizeof(const float *));
    float **outputs = (float **)alloca((size_t)nchan * sizeof(float *));
    int dspCount;

    for (int ch = 0; ch < nchan; ++ch) {
        inputs[ch] = buffer.getReadPointer(ch) + segstart;
        outputs[ch] = buffer.getWritePointer(ch) + segstart;
    }

    // downsample
    {
        WDL_ResampleSample *dsIn;
        impl->m_downsampler.SetFeedMode(true);
        int ret = impl->m_downsampler.ResamplePrepare(segsize, nchan, &dsIn);
        jassert(ret == segsize); (void)ret;
        interleave(inputs, dsIn, nchan, segsize);

        float *dsOut = impl->m_downsampler_out.data();
        int dsOutCap = impl->m_downsampler_out.size();
        dspCount = impl->m_downsampler.ResampleOut(dsOut, segsize, dsOutCap, nchan);

        deinterleave(dsOut, impl->m_dsp_in.getArrayOfWritePointers(), nchan, dspCount);
    }

    // DSP
    if (dspCount > 0) {
        processChipDSP(impl->m_dsp_in, impl->m_dsp_out, dspCount);

        for (int ch = 0; ch < nchan; ++ch)
            impl->m_last_dsp_outs.set(ch, outputs[ch][dspCount - 1]);
    }

    // upsample
    if (dspCount > 0) {
        WDL_ResampleSample *usIn;
        impl->m_upsampler.SetFeedMode(true);
        int ret = impl->m_upsampler.ResamplePrepare(dspCount, nchan, &usIn);
        jassert(ret == dspCount); (void)ret;

        interleave(impl->m_dsp_out.getArrayOfReadPointers(), usIn, nchan, dspCount);

        float *usOut = impl->m_upsampler_out.data();
        //int usOutCap = impl->m_upsampler_out.size();

        int finalCount = impl->m_upsampler.ResampleOut(usOut, dspCount, segsize, nchan);

        int pad = segsize - finalCount;
        for (int ch = 0; ch < nchan; ++ch)
            std::fill_n(outputs[ch], pad, impl->m_last_dsp_outs[ch]);

        float **outputsWithPad = (float **)alloca((size_t)nchan * sizeof(float *));
        for (int ch = 0; ch < nchan; ++ch)
            outputsWithPad[ch] = outputs[ch] + pad;

        deinterleave(usOut, outputsWithPad, nchan, finalCount);
    }
    else {
        for (int ch = 0; ch < nchan; ++ch)
            std::fill_n(outputs[ch], segsize, impl->m_last_dsp_outs[ch]);
    }
}

void BWProcessor::processChipDSP(const juce::AudioBuffer<float> &inputs, juce::AudioBuffer<float> &outputs, int nframes)
{
    Impl *impl = m_impl.get();
    int nchan = inputs.getNumChannels();

    float scale_db = impl->m_param.quant_scale->load(std::memory_order_relaxed);
    float scale_factor = std::pow(10.0f, scale_db / 20);
    float quant_bits = impl->m_param.quant_bits->load(std::memory_order_relaxed);
    int delta_speed = (int)impl->m_param.delta_speed->load(std::memory_order_relaxed);
    float delta_noise = impl->m_param.delta_noise->load(std::memory_order_relaxed);

    float quant_factor_from_16 = scale_factor * std::exp2(quant_bits - 1) / 32767.0f;
    float quant_factor_to_16 = 32767.0f / std::exp2(quant_bits - 1) / scale_factor;

    int delta_noise16b = juce::roundToInt(quant_factor_to_16 * delta_noise);

    for (int ch = 0; ch < nchan; ++ch) {
        const float *in = inputs.getReadPointer(ch);
        float *out = outputs.getWritePointer(ch);

        int sampleQ = impl->m_last_q_sample[ch];

        for (int i = 0; i < nframes; ++i) {
            float inp32f = in[i];

            // to 16-bit
            int inp16b = juce::jlimit(-32767, +32767, juce::roundToInt(inp32f * 32767.0f));

            // quantize down / scale
            int targetQ = juce::roundToInt((float)inp16b * quant_factor_from_16);

            // calculate the differential
            int delta = juce::jlimit(-delta_speed, +delta_speed, targetQ - sampleQ);
            //delta = delta ? delta : 1;

            // advance some steps up or down
            sampleQ += delta;

            // quantize up / unscale
            int out16b = juce::roundToInt((float)sampleQ * quant_factor_to_16);

            // if delta = 0, add delta static noise (upwards)
            out16b += delta ? 0 : delta_noise16b;
            // and pretend the signal moved up 1 step
            sampleQ += delta ? 0 : 1;

            // to single-float
            float out32f = (float)out16b / 32767.0f;

            out[i] = out32f;
        }

        impl->m_last_q_sample.set(ch, (int16_t)sampleQ);
    }
}

//==============================================================================
juce::AudioProcessorEditor *BWProcessor::createEditor()
{
    return nullptr;
}

bool BWProcessor::hasEditor() const
{
    return false;
}

//==============================================================================
const juce::String BWProcessor::getName() const
{
    return juce::CharPointer_UTF8{JucePlugin_Name};
}

bool BWProcessor::acceptsMidi() const
{
    return false;
}

bool BWProcessor::producesMidi() const
{
    return false;
}

bool BWProcessor::isMidiEffect() const
{
    return false;
}

double BWProcessor::getTailLengthSeconds() const
{
    return 0;
}

//==============================================================================
int BWProcessor::getNumPrograms()
{
    return 1;
}

int BWProcessor::getCurrentProgram()
{
    return 0;
}

void BWProcessor::setCurrentProgram(int index)
{
    (void)index;
}

const juce::String BWProcessor::getProgramName(int index)
{
    (void)index;
    return juce::String{};
}

void BWProcessor::changeProgramName(int index, const juce::String &newName)
{
    (void)index;
    (void)newName;
}

//==============================================================================
void BWProcessor::getStateInformation(juce::MemoryBlock &destData)
{
    Impl *impl = m_impl.get();

    juce::ValueTree state = impl->m_treestate->copyState();
    juce::MemoryOutputStream output{destData, false};
    state.writeToStream(output);
}

void BWProcessor::setStateInformation(const void *data, int sizeInBytes)
{
    Impl *impl = m_impl.get();

    juce::ValueTree state = juce::ValueTree::readFromData(data, (size_t)sizeInBytes);
    impl->m_treestate->replaceState(state);
}

//==========================================================================
bool BWProcessor::isBusesLayoutSupported(const BusesLayout &layout) const
{
    if (layout.inputBuses.size() != 1 || layout.outputBuses.size() != 1)
        return false;

    const juce::AudioChannelSet in = layout.getMainInputChannelSet();
    const juce::AudioChannelSet out = layout.getMainOutputChannelSet();

    return in == out &&
        (in == juce::AudioChannelSet::mono() ||
         in == juce::AudioChannelSet::stereo());
}

//==========================================================================

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter()
{
    return new BWProcessor;
}
