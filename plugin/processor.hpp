//Copyright CC) 2022- Jean Pierre Cimalando <jp-dev@gmx.com>
//SPDX-License-Identifier: GPL-3.0-or-later

#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <memory>

class BWProcessor final : public juce::AudioProcessor {
public:
    BWProcessor();
    ~BWProcessor() override;

    //==========================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float> &buffer, juce::MidiBuffer &midiMessages) override;
    void processBlock(juce::AudioBuffer<double> &buffer, juce::MidiBuffer &midiMessages) override;
    bool supportsDoublePrecisionProcessing() const override;

    void processSegment(juce::AudioBuffer<float> &buffer, int segstart, int segsize);
    void processChipDSP(const juce::AudioBuffer<float> &inputs, juce::AudioBuffer<float> &outputs, int nframes);

    //==========================================================================
    juce::AudioProcessorEditor *createEditor() override;
    bool hasEditor() const override;

    //==========================================================================
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==========================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String &newName) override;

    //==========================================================================
    void getStateInformation(juce::MemoryBlock &destData) override;
    void setStateInformation(const void *data, int sizeInBytes) override;

    //==========================================================================
    bool isBusesLayoutSupported(const BusesLayout &layout) const override;

private:
    //==========================================================================
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
