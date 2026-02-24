
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Power, Activity, BrainCircuit, FastForward, ListOrdered, Sparkles, Trash2, Repeat, Sun, Moon, ShieldAlert, LayoutList } from 'lucide-react';
import { Track, MasterState, DJMode, TransitionStyle } from './types';
import TrackList from './components/TrackList';
import DeckControls from './components/DeckControls';
import VuMeter from './components/VuMeter';
import { analyzeTrackHardwareDSP } from './services/audioAnalysisService';
import { FXType } from './components/FXRack';
import { sortSetFlow, suggestNextTrack, analyzeAudioWithGemini, getTransitionFXPlan, FXStep, generateTrackEmbeddings, analyzeTransitionCompatibility, selectTransitionStyle } from './services/intelligenceService';
import WelcomeScreen from './components/WelcomeScreen';
import { analysisQueue, AnalysisStatus } from './services/AnalysisQueue';
import { fetchStemBuffers, pollForHQStems, requestStemSeparation, checkStemServer, checkStemCache } from './services/stemService';

import { audioChain, DeckSignalChain } from './services/AudioSignalChain';
import { clock } from './services/Clock';
import { quantizer } from './services/Quantizer';
import { MixLogic } from './services/MixLogic';
import Visualizer from './components/Visualizer';
import RGBWaveform from './components/RGBWaveform';
import SimilarityMap from './components/SimilarityMap';
import ErrorBoundary from './components/ErrorBoundary';
import { telemetry, LogEntry } from './services/telemetry';
import { sampleSpectrum, computeSpectralMix, SpectralMixRecommendation } from './services/SpectralMixer';
import { LUFSMeter } from './services/LUFSMeter';

const DEFAULT_BPM = 128;

const App: React.FC = () => {
    const [hasStarted, setHasStarted] = useState(false);
    const [audioCtx, setAudioCtx] = useState<AudioContext | null>(null);
    const [tracks, setTracks] = useState<Track[]>([]);
    const [autoMixQueue, setAutoMixQueue] = useState<string[]>([]);

    // ... (rest of state)


    const [isPlaying, setIsPlaying] = useState(false);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [masterBpm, setMasterBpm] = useState(DEFAULT_BPM);
    const [deckPlayback, setDeckPlayback] = useState([{ current: 0, duration: 0, isPlaying: false }, { current: 0, duration: 0, isPlaying: false }]);
    const [deckSync, setDeckSync] = useState([true, true]);

    // State definitions were already present further down
    const [activeDeckIndex, setActiveDeckIndex] = useState(0);
    const [isTakeoverMode, setIsTakeoverMode] = useState(true);
    const [isAiProcessing, setIsAiProcessing] = useState(false);
    const [skipScheduled, setSkipScheduled] = useState(false);
    const [timeUntilTransition, setTimeUntilTransition] = useState<number | null>(null);
    const [theme, setTheme] = useState<'dark' | 'light'>('dark');
    const [creativity, setCreativity] = useState(0.5);
    const [isJoinDotsMode, setIsJoinDotsMode] = useState(false);
    const [dotWaypoints, setDotWaypoints] = useState<string[]>([]);
    const [activeTab, setActiveTab] = useState<'deckA' | 'mixer' | 'deckB'>('mixer');
    const [deckEQ, setDeckEQ] = useState([
        { low: 1, mid: 1, high: 1 },
        { low: 1, mid: 1, high: 1 }
    ]);
    const [trackWaveforms, setTrackWaveforms] = useState<Record<string, { data: number[][], structure: any[] }>>({});
    const [spectralCollision, setSpectralCollision] = useState(0);
    const [glitchIntensity, setGlitchIntensity] = useState(0); // 0-1
    const crossfaderRef = useRef(0); // Synced from masterState.crossfader for spectral mixer
    const [isMashupActive, setIsMashupActive] = useState(false);
    const lufsMeterRef = useRef(new LUFSMeter());
    const [lufsReading, setLufsReading] = useState({ momentary: -Infinity, truePeak: -Infinity, isClipping: false });

    // Glitch Decay Loop
    useEffect(() => {
        if (glitchIntensity > 0) {
            const id = requestAnimationFrame(() => {
                setGlitchIntensity(prev => Math.max(0, prev - 0.05));
            });
            return () => cancelAnimationFrame(id);
        }
    }, [glitchIntensity]);

    const triggerGlitch = (amount: number = 1.0) => {
        setGlitchIntensity(amount);
        if (amount > 0.5) telemetry.warn("SYSTEM_INSTABILITY_DETECTED");
    };

    // ── Spectral Collision Logic (Isolator Protection + RGB Bass-Swap) ─────────
    const lastSpectralStrategyRef = useRef<string>('NONE');
    const checkSpectralCollision = useCallback(() => {
        const d1 = decksRef.current[0];
        const d2 = decksRef.current[1];
        const p1Data = deckPlayback[0];
        const p2Data = deckPlayback[1];

        if (!d1.active || !d2.active || !d1.isPlaying || !d2.isPlaying) {
            setSpectralCollision(0);
            return;
        }

        const w1 = d1.trackId ? trackWaveforms[d1.trackId] : null;
        const w2 = d2.trackId ? trackWaveforms[d2.trackId] : null;
        if (!w1 || !w2) {
            setSpectralCollision(0);
            return;
        }

        const dur1 = p1Data.duration;
        const dur2 = p2Data.duration;
        if (dur1 <= 0 || dur2 <= 0) return;

        // Sample RGB spectrum at current playback positions
        const specA = sampleSpectrum(w1.data, p1Data.current, dur1);
        const specB = sampleSpectrum(w2.data, p2Data.current, dur2);

        // Compute spectral mix recommendation
        const mix = computeSpectralMix(specA, specB, crossfaderRef.current);
        setSpectralCollision(mix.collisionSeverity);

        // ── Apply RGB-Aware EQ Automation ──
        // Only auto-adjust when both decks are actively playing
        if (audioChain && mix.strategy !== 'NONE') {
            const chain = audioChain;
            const ctx = chain.ctx;
            const smooth = 0.15; // 150ms smoothing to avoid zipper noise

            // Bass: Surgically cut the outgoing deck's bass
            chain.decks[0].strips.bass.gain.gain.setTargetAtTime(mix.deckABassGain, ctx.currentTime, smooth);
            chain.decks[1].strips.bass.gain.gain.setTargetAtTime(mix.deckBBassGain, ctx.currentTime, smooth);

            // Mid: Carve if needed
            if (mix.strategy === 'MID_CARVE' || mix.strategy === 'FULL_SPECTRAL') {
                chain.decks[0].strips.other.gain.gain.setTargetAtTime(mix.deckAMidGain, ctx.currentTime, smooth);
                chain.decks[1].strips.other.gain.gain.setTargetAtTime(mix.deckBMidGain, ctx.currentTime, smooth);
            }

            // Log strategy changes to telemetry (throttled)
            if (mix.strategy !== lastSpectralStrategyRef.current) {
                lastSpectralStrategyRef.current = mix.strategy;
                telemetry.ai(`SPECTRAL: ${mix.strategy} (COLLISION: ${(mix.collisionSeverity * 100).toFixed(0)}%)`);
            }
        }
    }, [trackWaveforms, deckPlayback]);

    const registerTrackWithSimilarityServer = async (track: Track) => {
        if (!track.embedding || !track.artistEmbedding) return;
        try {
            await fetch('http://localhost:8000/similarity/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    track_id: track.id,
                    embedding: track.embedding,
                    artist_embedding: track.artistEmbedding,
                    key: track.key
                })
            });
        } catch (e) {
            console.error("Failed to register track with similarity server:", e);
        }
    };

    const generateJoinDotsPlaylist = async () => {
        if (dotWaypoints.length < 2) return;
        telemetry.ai("NEURAL: JOINING THE DOTS (HYPER-COMPUTE)...");
        const fullPath: string[] = [];

        // Parallel fetch segments for maximum efficiency
        const segmentPromises = dotWaypoints.slice(0, -1).map((wp, i) =>
            fetch(`http://localhost:8000/similarity/path?start_id=${wp}&end_id=${dotWaypoints[i + 1]}&intermediate_count=2&creativity=${creativity}&noise=${creativity * 0.3}`)
                .then(r => r.json())
                .catch(() => ({ path: [wp, dotWaypoints[i + 1]] }))
        );

        const segments = await Promise.all(segmentPromises);
        segments.forEach((data, i) => {
            // Add everything except the last one of each segment (it's the start of the next or the final end)
            fullPath.push(...(data.path.slice(0, -1)));
        });

        fullPath.push(dotWaypoints[dotWaypoints.length - 1]);
        setAutoMixQueue(fullPath);
        telemetry.success(`SEQUENCE SYNTHESIZED: ${fullPath.length} NODES`);
        setIsJoinDotsMode(false);
        setDotWaypoints([]);
    };

    const [masterState, setMasterState] = useState<MasterState>({
        isRecording: false,
        recordingTime: 0,
        crowdVolume: 0.1,
        aiHostEnabled: false,
        limiterReduction: 0,
        mode: DJMode.AUTO,
        crossfader: -1,
        masterFxType: 'NONE',
        masterFxWet: 0
    });

    const [energyScore, setEnergyScore] = useState(5.0);
    const [phraseCountdown, setPhraseCountdown] = useState(32);
    const mixLogic = useMemo(() => new MixLogic(audioChain), []);

    // ── Musical Engine: High-Frequency Ticker ───────────────────────────
    useEffect(() => {
        let frameId: number;

        const loop = (time: number) => {
            const dt = (time - clock.lastTime) / 1000;
            clock.tick(dt);
            clock.lastTime = time;

            // Update Musical State
            if (clock.isPlaying) {
                // Phrase Countdown: How many beats left in current 32-beat phrase
                const currentBeatInPhrase = (clock.barInPhrase * 4) + clock.beatInBar;
                setPhraseCountdown(32 - currentBeatInPhrase);

                // Dynamic Headroom: Call every frame to react to fader moves
                audioChain.applyHeadroomManagement();
            }

            frameId = requestAnimationFrame(loop);
        };

        if (hasStarted) {
            frameId = requestAnimationFrame(loop);
        }
        return () => cancelAnimationFrame(frameId);
    }, [hasStarted]);

    const [deckStems, setDeckStems] = useState([
        { vocal: 1, instru: 1, bass: 1, kick: 1, hihat: 1, stemFx: 0 },
        { vocal: 1, instru: 1, bass: 1, kick: 1, hihat: 1, stemFx: 0 }
    ]);
    const [deckFilters, setDeckFilters] = useState<number[]>([0.5, 0.5]);
    const [deckLoops, setDeckLoops] = useState([{ active: false, length: 4 }, { active: false, length: 4 }]);
    const [deckFX, setDeckFX] = useState<Record<FXType, { active: boolean; value: number }>[]>(
        [
            { ECHO: { active: false, value: 0.5 }, REVERB: { active: false, value: 0.4 }, CRUSH: { active: false, value: 0 }, WASH: { active: false, value: 0 } },
            { ECHO: { active: false, value: 0.5 }, REVERB: { active: false, value: 0.4 }, CRUSH: { active: false, value: 0 }, WASH: { active: false, value: 0 } }
        ]
    );


    // State for Deck Metadata
    interface DeckMetadata {
        trackId: string | null;
        startTime: number;
        pauseTime: number | null;
        playbackRate: number;
        isPlaying: boolean;
        duration: number;
        active: boolean;
        source?: AudioBufferSourceNode;
        analyser?: AnalyserNode;
        usingStemMode: boolean;
        globalFilter?: number;
    }

    const masterRef = useRef<any>(null);
    const decksRef = useRef<DeckMetadata[]>([
        { trackId: null, startTime: 0, pauseTime: null, playbackRate: 1, isPlaying: false, duration: 0, active: false, globalFilter: 0.5, usingStemMode: false },
        { trackId: null, startTime: 0, pauseTime: null, playbackRate: 1, isPlaying: false, duration: 0, active: false, globalFilter: 0.5, usingStemMode: false }
    ]);
    const isAutoMixingRef = useRef<boolean>(false);
    const mixLogicRef = useRef<MixLogic>(new MixLogic(audioChain));
    const tracksRef = useRef<Track[]>([]);
    const playHistoryRef = useRef<Track[]>([]); // Deej-AI lookback window
    const [stemServerAvailable, setStemServerAvailable] = useState(false);
    // 30-second lookahead: pre-computed analysis so transition fires instantly
    const lookaheadRef = useRef<{ nextTrack: Track; style: TransitionStyle; rmsBoostDb: number } | null>(null);
    const lookaheadDoneRef = useRef<boolean>(false);

    const lastAnalyzedCountRef = useRef<number>(0);

    useEffect(() => { tracksRef.current = tracks; }, [tracks]);

    // ── Autonomous Sequence & Playback Handler ──────────────────────────────────
    useEffect(() => {
        if (tracks.length === 0) return;

        const validTracks = tracks.filter(t => t.embedding);
        const allAnalyzed = tracks.every(t => t.embedding || t.status === 'error');

        if (allAnalyzed && validTracks.length > 0 && validTracks.length > lastAnalyzedCountRef.current && !isAiProcessing) {
            const autoConfigureSet = async () => {
                setIsAiProcessing(true);
                telemetry.ai("NEW STEMS DETECTED. RE-CALCULATING OPTIMAL FLOW...");

                try {
                    const res = await sortSetFlow(tracks);
                    lastAnalyzedCountRef.current = validTracks.length;

                    if (res && res.length > 0) {
                        // If engine is rolling but decks are stopped, auto-start
                        if (!isPlaying && hasStarted) {
                            setAutoMixQueue(res.slice(1)); // Queue the rest
                            setTimeout(() => {
                                telemetry.ai("AUTONOMOUS PLAYBACK INITIATED");
                                loadTrackToDeck(res[0], 0, true);
                            }, 500);
                        } else {
                            setAutoMixQueue(res);
                            telemetry.success("AUTO SEQUENCE MAP UPDATED");
                        }
                    }
                } catch (e) {
                    console.error("Auto Flow Error:", e);
                } finally {
                    setIsAiProcessing(false);
                }
            };

            autoConfigureSet();
        }
    }, [tracks, isAiProcessing, isPlaying, hasStarted]);

    useEffect(() => {
        if (theme === 'light') document.body.classList.add('light-mode');
        else document.body.classList.remove('light-mode');
    }, [theme]);

    // ── Audio Initialization ─────────────────────────────────────────────────────
    // Browsers block AudioContext until a user gesture.
    // We must show a "Click to Start" overlay.
    const initializeAudio = async () => {
        if (hasStarted) return;

        // 1. Resume Context
        if (audioChain.ctx.state === 'suspended') {
            await audioChain.ctx.resume();
        }
        setAudioCtx(audioChain.ctx);

        // 2. Start Global Clock
        clock.isPlaying = true;
        clock.lastTime = performance.now(); // Reset delta tracking

        // 3. Update State
        setHasStarted(true);
        telemetry.info("SYSTEM: AUDIO ENGINE ONLINE");
    };

    // Subscribe to Telemetry Service
    useEffect(() => {
        const unsubscribe = telemetry.subscribe((entry) => {
            setLogs(prev => [entry, ...prev].slice(0, 50));
        });
        return () => unsubscribe();
    }, []);

    // Initial check
    useEffect(() => {
        checkStemServer().then(available => {
            setStemServerAvailable(available);
            if (available) telemetry.success('STEM_SERVER: ONLINE');
        });
    }, []);

    const getSmoothTime = (ratio = 0.125) => (60 / masterBpm) * ratio;

    useEffect(() => {
        if (!audioCtx) return;
        decksRef.current.forEach((deck, i) => {
            if (deck.source && deck.active) {
                const track = tracksRef.current.find(t => t.id === deck.trackId);
                const trackBpm = track?.detectedBpm || DEFAULT_BPM;
                const newRate = deckSync[i] ? (masterBpm / trackBpm) : 1;
                audioChain.setPlaybackRate(i, newRate);
                deck.playbackRate = newRate;
            }
        });
    }, [masterBpm, deckSync, audioCtx]);

    useEffect(() => {
        if (!audioCtx) return;

        let animationFrameId: number;
        let lastTime = performance.now();

        const tick = () => {
            const now = performance.now();
            const deltaTime = (now - lastTime) / 1000;
            lastTime = now;

            // 1. Update Global Clock
            clock.tick(deltaTime);

            // 2. Synchronize Decks (PLL)
            if (audioCtx) {
                decksRef.current.forEach((deck, i) => {
                    if (deck.active && deck.isPlaying && deck.playbackRate > 0 && deckSync[i]) {
                        const track = tracksRef.current.find(t => t.id === deck.trackId);
                        if (track) {
                            const trackBpm = track.detectedBpm || DEFAULT_BPM;
                            const nominalRate = masterBpm / trackBpm;

                            // Calculate Deck Phase
                            // Time elapsed since start (adjusted for rate changes would be complex, 
                            // but for steady state: Time * Bps)
                            // A better measure is: (AudioContextTime - StartTime) * (TrackBps)
                            // But StartTime needs to be adjusted when Rate changes... 
                            // For this MVP, we use the simple linear time approximation which works if rate is mostly constant.
                            // Ideally we'd read the Time from the AudioSourceNode if it supported it.

                            const currentDeckTime = audioCtx.currentTime - deck.startTime;
                            // We need to account for the integrated rate over time if it changes often.
                            // But for "Lock", we just want the beat grid alignment.

                            const beatPosition = currentDeckTime * (trackBpm / 60);
                            const deckPhase = beatPosition % 1.0;

                            // Get Correction from Clock
                            // Only apply if we are close to the master bpm (sync engaged)
                            // and not scratching/seeking.
                            const correction = clock.getSyncCorrection(deckPhase);

                            // Apply Nudge
                            const targetRate = nominalRate * correction;

                            // Smooth update to avoid zipper noise, but fast enough for sync
                            audioChain.setPlaybackRate(i, targetRate);
                            deck.playbackRate = targetRate;
                        }
                    }
                });
            }

            // 3. Update UI State (throttled to ~30fps to save React renders if needed, or run full speed)
            // For smoothly animating visualizations, 60fps is good.
            setDeckPlayback(prev => {
                const next = [...prev];
                audioChain.decks.forEach((audioDeck, i) => {
                    // Update deck state from Signal Chain
                    // Simplification: Time reference
                    const src = audioDeck.bufferASource || (audioDeck.bufferBSources ? audioDeck.bufferBSources.vocals : null);
                    if (src && src.buffer) {
                        // Estimate current time
                        const deck = decksRef.current[i];
                        const estimatedTime = deck.active ? (audioCtx.currentTime - deck.startTime) * deck.playbackRate : deck.pauseTime || 0;
                        next[i] = { ...next[i], isPlaying: deck.isPlaying, duration: src.buffer.duration, current: estimatedTime };
                    } else {
                        next[i] = { ...next[i], isPlaying: false, current: 0 };
                    }
                });
                return next;
            });

            // 4. Heavy Analysis (Decoupled from 60fps)
            // Moved to setInterval (10Hz) below

            // 5. Continuation
            animationFrameId = requestAnimationFrame(tick);
        };

        const analyticsTick = () => {
            // Run heavy math at 10Hz
            checkSpectralCollision();

            // LUFS Measurement (Safety Warden)
            if (masterRef.current?.analyser) {
                const reading = lufsMeterRef.current.measure(masterRef.current.analyser);
                setLufsReading({ momentary: reading.momentary, truePeak: reading.truePeak, isClipping: reading.isClipping });

                // Anti-Clipping: If true peak exceeds -1dBTP, trigger gain reduction
                if (reading.isClipping) {
                    audioChain.masterBus.gain.setTargetAtTime(
                        Math.max(0.5, audioChain.masterBus.gain.value * 0.95),
                        audioChain.ctx.currentTime, 0.01
                    );
                }
            }

            // Headroom Management
            audioChain.applyHeadroomManagement();

            // Autonomous Logic (Beat counting, phrase detection)
            if (isTakeoverMode && audioCtx) {
                const activeDeckIdx = decksRef.current[0].active ? 0 : 1;
                const deck = decksRef.current[activeDeckIdx];
                const track = tracksRef.current.find(t => t.id === deck.trackId);
                if (deck.isPlaying && track) {
                    const currentTime = deck.isPlaying ? (audioCtx.currentTime - deck.startTime) * deck.playbackRate : deck.pauseTime || 0;
                    const remainingTime = track.duration - currentTime;

                    if (remainingTime < 30 && remainingTime > 29) runLookahead(activeDeckIdx);

                    // Update UI countdown
                    if (remainingTime <= 30 && remainingTime >= 0) {
                        const secondsLeft = Math.floor(remainingTime);
                        setTimeUntilTransition(secondsLeft);
                    } else {
                        setTimeUntilTransition(null);
                    }

                    if ((remainingTime < 2 && remainingTime > 0) || skipScheduled) {
                        setSkipScheduled(false);
                        triggerAiTakeoverMix(activeDeckIdx, skipScheduled);
                    }
                }
            }
        };

        animationFrameId = requestAnimationFrame(tick);
        const intervalId = setInterval(analyticsTick, 100);

        return () => {
            cancelAnimationFrame(animationFrameId);
            clearInterval(intervalId);
        };
    }, [audioCtx, isTakeoverMode, skipScheduled, masterBpm, deckSync, checkSpectralCollision]);

    // ── 30-Second Lookahead ────────────────────────────────────────────────────
    // Pre-fetches the next track and runs transition analysis so the actual
    // transition fires instantly without any async delay.
    const runLookahead = useCallback(async (currentDeckIdx: number) => {
        if (lookaheadDoneRef.current || isAutoMixingRef.current) return;
        lookaheadDoneRef.current = true;

        const currentTrack = tracksRef.current.find(t => t.id === decksRef.current[currentDeckIdx]?.trackId);
        if (!currentTrack) return;

        telemetry.ai('LOOKAHEAD: PRE-COMPUTING TRANSITION...');

        let nextTrack: Track | undefined;
        const nextTrackId = autoMixQueue[0];
        if (nextTrackId) {
            nextTrack = tracksRef.current.find(t => t.id === nextTrackId);
        } else {
            nextTrack = await suggestNextTrack(currentTrack, tracksRef.current.filter(t => t.id !== currentTrack.id), playHistoryRef.current.slice(-3));
        }
        if (!nextTrack) return;

        const analysis = analyzeTransitionCompatibility(currentTrack, nextTrack);
        const style = selectTransitionStyle(analysis, currentTrack, nextTrack);
        const rmsBoostDb = Math.max(0, Math.min(analysis.rmsRatioDb, 6)); // cap boost at +6dB

        lookaheadRef.current = { nextTrack, style, rmsBoostDb };
        telemetry.ai(`LOOKAHEAD: ${style} → ${nextTrack.name} (RMS+${rmsBoostDb.toFixed(1)}dB)`);
    }, [autoMixQueue]);


    // ── Main Transition Executor ─────────────────────────────────────────────────
    const triggerAiTakeoverMix = async (currentDeckIdx: number, isFastSkip: boolean = false) => {
        if (isAutoMixingRef.current || !audioCtx) return;
        isAutoMixingRef.current = true;
        setTimeUntilTransition(0);

        const nextDeckIdx = currentDeckIdx === 0 ? 1 : 0;
        const currentTrack = tracksRef.current.find(t => t.id === decksRef.current[currentDeckIdx].trackId);

        // ── Stage 0: Resolve next track (use lookahead if available) ──────────
        let nextTrack: Track | undefined;
        let style: TransitionStyle = 'ECHO_OUT';
        let rmsBoostDb = 0;

        if (lookaheadRef.current && !isFastSkip) {
            ({ nextTrack, style, rmsBoostDb } = lookaheadRef.current);
            lookaheadRef.current = null;
        } else {
            const nextTrackId = autoMixQueue[0];
            if (nextTrackId) {
                nextTrack = tracksRef.current.find(t => t.id === nextTrackId);
                setAutoMixQueue(prev => prev.slice(1));
            } else {
                telemetry.ai('NEURAL: GENERATING SONIC SEQUENCE...');
                try {
                    nextTrack = await suggestNextTrack(currentTrack, tracksRef.current.filter(t => t.id !== currentTrack?.id), playHistoryRef.current.slice(-3));
                } catch (e) {
                    telemetry.error("NEURAL FAILURE: COULD NOT COMPUTE NEXT TRACK");
                    nextTrack = undefined;
                }
            }
            if (nextTrack && currentTrack) {
                const analysis = analyzeTransitionCompatibility(currentTrack, nextTrack);
                style = selectTransitionStyle(analysis, currentTrack, nextTrack);
                rmsBoostDb = Math.max(0, Math.min(analysis.rmsRatioDb, 6));
            }
        }
        lookaheadDoneRef.current = false;

        if (!nextTrack) {
            telemetry.success('AI_DJ: PERFORMANCE FINISHED.');
            // Stop the current deck since there is nowhere to transition to
            audioChain.decks[currentDeckIdx].stereoFader.gain.setTargetAtTime(0, audioCtx.currentTime, 0.5);
            setTimeout(() => {
                const stopDeck = (di: number) => {
                    audioChain.stopDeck(di);
                    setDeckPlayback(p => { const o = [...p]; o[di] = { ...o[di], isPlaying: false, current: 0 }; return o; });
                    decksRef.current[di].isPlaying = false;
                    decksRef.current[di].pauseTime = null;
                };
                stopDeck(currentDeckIdx);
                isAutoMixingRef.current = false;
            }, 600);
            return;
        }

        // Force LOOP_ROLL if deck loop is active or fast-skip requested
        if (deckLoops[currentDeckIdx].active || isFastSkip) style = 'LOOP_ROLL';

        // ── Stem-powered transition upgrade ───────────────────────────────────
        const stemsAvailableA = !!(currentTrack?.stemsReady && currentTrack?.stemBuffers);
        const stemsAvailableB = !!(nextTrack.stemsReady && nextTrack.stemBuffers);
        const canUseStemTransition = stemsAvailableA && stemsAvailableB;

        // Upgrade to stem-powered variant when stems are ready
        // Ghost Mashup: harmonic match → keep A vocals, swap drums/bass to B
        // Percussive Bridge: incompatible key → mute all B except drums first
        let stemTransitionMode: 'GHOST_MASHUP' | 'PERCUSSIVE_BRIDGE' | 'STEM_WASH' | null = null;
        if (canUseStemTransition) {
            const analysis = analyzeTransitionCompatibility(currentTrack!, nextTrack);
            if (analysis.camelotScore >= 0.8 && style === 'EQ_SWAP') {
                stemTransitionMode = 'GHOST_MASHUP';
            } else if (analysis.camelotScore < 0.55 && style === 'FILTER_SWEEP') {
                stemTransitionMode = 'PERCUSSIVE_BRIDGE';
            } else if (style === 'REVERB_WASH') {
                stemTransitionMode = 'STEM_WASH';
            }
            if (stemTransitionMode) telemetry.ai(`STEM_MODE: ${stemTransitionMode}`);
        }

        // ── Stage 1: Pre-Sync ─────────────────────────────────────────────────
        const startBpm = masterBpm;
        const targetBpm = nextTrack.detectedBpm || DEFAULT_BPM;
        const gainBoostLinear = rmsBoostDb > 0 ? Math.pow(10, rmsBoostDb / 20) : 1;

        const durationMap: Record<TransitionStyle, number> = {
            PHRASES_CROSSFADE: 32, EQ_SWAP: 16, FILTER_SWEEP: 16,
            ECHO_OUT: 12, LOOP_ROLL: 16, BACKSPIN: 3,
            REVERB_WASH: 16, BRAKE_STOP: 6, SLAM: 0.01, GATER: 16, DOUBLE_DROP: 16
        };
        const mixDuration = durationMap[style] ?? 16;

        const forcedInStart = (style === 'SLAM')
            ? (nextTrack.structure?.drops?.[0] || 0)
            : Math.max(nextTrack.structure?.musicStartTime || 0, (nextTrack.structure?.drops?.[0] || 0) - mixDuration);

        await loadTrackToDeck(nextTrack.id, nextDeckIdx, true, forcedInStart);

        // Gain compensation: boost B if it's quieter than A
        if (gainBoostLinear > 1) {
            audioChain.decks[nextDeckIdx].stereoFader.gain.setTargetAtTime(
                Math.min(gainBoostLinear, 2), audioCtx.currentTime, 0.1
            );
        }

        telemetry.ai(`MIX: ${style}${stemTransitionMode ? `+${stemTransitionMode}` : ''} → ${nextTrack.name}`);

        // ── Stage 2: Frequency Carve ──────────────────────────────────────────
        // Vocal protector: blur A's mids if both tracks have vocals
        const hasVocalsA = !!(currentTrack?.genre && !['techno', 'minimal', 'ambient', 'instrumental'].some(g => (currentTrack.genre || '').toLowerCase().includes(g)));
        const hasVocalsB = !!(nextTrack.genre && !['techno', 'minimal', 'ambient', 'instrumental'].some(g => (nextTrack.genre || '').toLowerCase().includes(g)));
        if (hasVocalsA && hasVocalsB) {
            audioChain.decks[currentDeckIdx].fx.reverbGain.gain.setTargetAtTime(0.15, audioCtx.currentTime, 0.3);
            telemetry.ai('VOCAL_PROTECTOR: MID-CARVE ACTIVE');
        }

        // ── Stage 3: Primary Action ───────────────────────────────────────────
        const fxPlan = await getTransitionFXPlan(currentTrack, nextTrack, style, mixDuration);
        const appliedSteps = new Set<number>();
        const useExpFader = ['SLAM', 'BACKSPIN', 'BRAKE_STOP'].includes(style);

        const barDuration = (60 / masterBpm) * 4;
        const startBarTime = audioCtx.currentTime + 0.5;
        const usingProLogic = canUseStemTransition && ['PHRASES_CROSSFADE', 'EQ_SWAP', 'GHOST_MASHUP', 'PERCUSSIVE_BRIDGE', 'DOUBLE_DROP', 'SLAM'].includes(style + (stemTransitionMode || ''));

        if (usingProLogic) {
            if (stemTransitionMode === 'GHOST_MASHUP') {
                mixLogic.executeGhostMashup(currentDeckIdx, nextDeckIdx, startBarTime, mixDuration / 4);
                telemetry.success("NEURAL_MIX: GHOST MASHUP ENGAGED");
            } else if (stemTransitionMode === 'PERCUSSIVE_BRIDGE') {
                mixLogic.executePercussiveBridge(currentDeckIdx, nextDeckIdx, startBarTime, mixDuration / 4);
                telemetry.success("NEURAL_MIX: PERCUSSIVE BRIDGE ENGAGED");
            } else if (style === 'DOUBLE_DROP') {
                mixLogic.executeDoubleDrop(currentDeckIdx, nextDeckIdx, startBarTime, mixDuration / 4);
                telemetry.success("NEURAL_MIX: DOUBLE DROP ENGAGED");
            } else if (style === 'SLAM') {
                mixLogic.executeBpmSlam(currentDeckIdx, nextDeckIdx, audioCtx.currentTime + 0.1);
                telemetry.success("NEURAL_MIX: BPM SLAM ENGAGED");
            } else {
                mixLogic.executeProTransition(currentDeckIdx, nextDeckIdx, startBarTime, mixDuration / 4);
                telemetry.success("NEURAL_MIX: SEQUENCER ENGAGED (SAMPLE-ACCURATE)");
            }
        } else {
            // Legacy/Fallback Logic
            const deckA = decksRef.current[currentDeckIdx];
            const deckB = decksRef.current[nextDeckIdx];

            if (style === 'FILTER_SWEEP' || style === 'REVERB_WASH') {
                const f = audioChain.decks[currentDeckIdx].fx.djFilter;
                f.type = style === 'FILTER_SWEEP' ? 'highpass' : 'lowpass';
                f.frequency.setValueAtTime(style === 'FILTER_SWEEP' ? 20 : 20000, audioCtx.currentTime);
                f.frequency.exponentialRampToValueAtTime(style === 'FILTER_SWEEP' ? 10000 : 200, audioCtx.currentTime + mixDuration);
            }
            if (style === 'BRAKE_STOP') audioChain.applyBrakeStop(currentDeckIdx);
            if (style === 'BACKSPIN') {
                audioChain.applyBrakeStop(currentDeckIdx);
                const f = audioChain.decks[currentDeckIdx].fx.djFilter;
                f.type = 'highpass';
                f.frequency.setValueAtTime(200, audioCtx.currentTime);
                f.frequency.exponentialRampToValueAtTime(12000, audioCtx.currentTime + mixDuration * 0.5);
            }

            // Stem-powered pre-actions
            if (stemTransitionMode === 'GHOST_MASHUP') {
                audioChain.decks[nextDeckIdx].strips.vocals.gain.gain.setTargetAtTime(0, audioCtx.currentTime, 0.05);
                telemetry.ai('GHOST_MASHUP: B VOCALS MUTED — A VOCALS LEADING');
            }
            if (stemTransitionMode === 'PERCUSSIVE_BRIDGE') {
                const acD = audioChain.decks[currentDeckIdx];
                acD.fx.echoGain.gain.setTargetAtTime(1, audioCtx.currentTime, getSmoothTime());
                acD.fx.echo.delayTime.setTargetAtTime(0.3, audioCtx.currentTime, getSmoothTime());
                acD.fx.washFilter.frequency.setTargetAtTime(2000, audioCtx.currentTime, getSmoothTime());
                audioChain.decks[nextDeckIdx].strips.other.gain.gain.setTargetAtTime(0, audioCtx.currentTime, 0.05);
                telemetry.ai('PERCUSSIVE_BRIDGE: B DRUMS ONLY');
            }
            if (stemTransitionMode === 'STEM_WASH') {
                audioChain.decks[currentDeckIdx].fx.reverbGain.gain.setTargetAtTime(0.8, audioCtx.currentTime, 0.5);
            }
            if (style === 'EQ_SWAP' && canUseStemTransition && !stemTransitionMode) {
                audioChain.decks[nextDeckIdx].strips.bass.gain.gain.setTargetAtTime(0, audioCtx.currentTime, 0.05);
            }

            // Stem Mixer Table (Legacy Timing)
            if (canUseStemTransition && audioCtx && ['EQ_SWAP', 'PHRASES_CROSSFADE', 'GHOST_MASHUP'].includes(style + (stemTransitionMode || ''))) {
                const ctx = audioCtx;
                const barMs = (60 / startBpm) * 4 * 1000;
                setTimeout(() => {
                    if (stemTransitionMode === 'GHOST_MASHUP') {
                        audioChain.decks[nextDeckIdx].strips.drums.gain.gain.setTargetAtTime(0.7, ctx.currentTime, 0.3);
                        audioChain.decks[nextDeckIdx].strips.bass.gain.gain.setTargetAtTime(0.7, ctx.currentTime, 0.3);
                    } else {
                        audioChain.decks[nextDeckIdx].strips.vocals.gain.gain.setTargetAtTime(0.5, ctx.currentTime, 0.3);
                        audioChain.decks[nextDeckIdx].strips.other.gain.gain.setTargetAtTime(0.5, ctx.currentTime, 0.3);
                    }
                }, barMs * 4);

                setTimeout(() => {
                    if (stemTransitionMode === 'GHOST_MASHUP') {
                        audioChain.decks[currentDeckIdx].strips.vocals.gain.gain.setTargetAtTime(0, ctx.currentTime, 0.5);
                        audioChain.decks[nextDeckIdx].strips.vocals.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.5);
                    } else {
                        audioChain.decks[nextDeckIdx].strips.vocals.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.3);
                        audioChain.decks[currentDeckIdx].strips.drums.gain.gain.setTargetAtTime(0.5, ctx.currentTime, 0.3);
                    }

                    if (style === 'EQ_SWAP') {
                        audioChain.decks[currentDeckIdx].strips.bass.gain.gain.setTargetAtTime(0, ctx.currentTime, 0.02);
                        audioChain.decks[nextDeckIdx].strips.bass.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.02);
                    }
                    if (stemTransitionMode === 'PERCUSSIVE_BRIDGE') {
                        audioChain.decks[nextDeckIdx].strips.other.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.3);
                        audioChain.decks[nextDeckIdx].strips.bass.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.3);
                    }
                    audioChain.decks[currentDeckIdx].strips.other.gain.gain.setTargetAtTime(0.7, ctx.currentTime, 0.2);

                }, barMs * 8);

                setTimeout(() => {
                    const dA = audioChain.decks[currentDeckIdx];
                    const dB = audioChain.decks[nextDeckIdx];
                    // Mute A all
                    dA.strips.vocals.gain.gain.setTargetAtTime(0, ctx.currentTime, 0.3);
                    dA.strips.drums.gain.gain.setTargetAtTime(0, ctx.currentTime, 0.3);
                    dA.strips.bass.gain.gain.setTargetAtTime(0, ctx.currentTime, 0.3);
                    dA.strips.other.gain.gain.setTargetAtTime(0, ctx.currentTime, 0.3);

                    // Full B
                    dB.strips.vocals.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.3);
                    dB.strips.drums.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.3);
                    dB.strips.bass.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.3);
                    dB.strips.other.gain.gain.setTargetAtTime(1, ctx.currentTime, 0.3);

                    telemetry.ai('STEM_MIX: FULL B');
                }, barMs * 12);
            }
        }

        // ── Tick loop: crossfader + BPM interpolation + FX steps ─────────────
        let progress = 0;
        const steps = Math.max(mixDuration * 10, 10);
        const tickTime = (mixDuration * 1000) / steps;
        let tickCount = 0;

        const interval = setInterval(() => {
            progress += (1 / steps);
            tickCount++;
            if (audioCtx) {
                // Smart-fader curve
                let cfVal: number;
                if (useExpFader) {
                    const t = Math.min(progress, 1);
                    const expP = t < 0.8 ? Math.pow(t / 0.8, 3) * 0.5 : 0.5 + (t - 0.8) / 0.2 * 0.5;
                    cfVal = currentDeckIdx === 0 ? -1 + expP * 2 : 1 - expP * 2;
                } else {
                    cfVal = currentDeckIdx === 0 ? -1 + progress * 2 : 1 - progress * 2;
                }

                // Constant-power crossfade
                const gainA = Math.cos((cfVal + 1) * 0.25 * Math.PI);
                const gainB = Math.sin((cfVal + 1) * 0.25 * Math.PI);

                // Only touch stereo fader if NOT using ProLogic scheduled mixing
                if (!usingProLogic) {
                    audioChain.decks[0].stereoFader.gain.setTargetAtTime(gainA, audioCtx.currentTime, 0.05);
                    audioChain.decks[1].stereoFader.gain.setTargetAtTime(gainB, audioCtx.currentTime, 0.05);
                }

                // BPM interpolation
                const currentBpm = startBpm + (targetBpm - startBpm) * progress;
                decksRef.current.forEach((deck, i) => {
                    if (deck.active) {
                        const track = tracksRef.current.find(t => t.id === deck.trackId);
                        const trackBpm = track?.detectedBpm || DEFAULT_BPM;
                        const rate = currentBpm / trackBpm;
                        audioChain.setPlaybackRate(i, rate);
                        deck.playbackRate = rate;
                    }
                });

                if (tickCount % 10 === 0) {
                    setMasterBpm(currentBpm);
                    setMasterState(prev => ({ ...prev, crossfader: cfVal }));
                }

                // Apply AI FX automation steps
                fxPlan.forEach((step, idx) => {
                    if (progress >= step.at && !appliedSteps.has(idx)) {
                        appliedSteps.add(idx);
                        const deckIdx = step.deck === 'A' ? currentDeckIdx : nextDeckIdx;
                        const acD = audioChain.decks[deckIdx];
                        const fxType = step.fx as FXType;
                        if (fxType === 'ECHO') {
                            acD.fx.echoGain.gain.setTargetAtTime(step.active ? step.value * 1.2 : 0, audioCtx.currentTime, 0.05);
                            acD.fx.echo.delayTime.setTargetAtTime((60 / currentBpm) * 0.75, audioCtx.currentTime, 0.05);
                        } else if (fxType === 'REVERB') {
                            acD.fx.reverbGain.gain.setTargetAtTime(step.active ? step.value : 0, audioCtx.currentTime, 0.05);
                        } else if (fxType === 'CRUSH') {
                            const curve = new Float32Array(44100);
                            const deg = step.active ? step.value * 80 : 0;
                            for (let i = 0; i < 44100; i++) {
                                const x = (i * 2) / 44100 - 1;
                                curve[i] = ((3 + deg) * x * 20 * (Math.PI / 180)) / (Math.PI + deg * Math.abs(x));
                            }
                            acD.fx.crush.curve = curve;
                        } else if (fxType === 'WASH') {
                            if (step.active) {
                                acD.fx.washFilter.frequency.setTargetAtTime(150 + (step.value * 17000), audioCtx.currentTime, 0.1);
                                acD.fx.washFilter.Q.setTargetAtTime(15, audioCtx.currentTime, 0.1);
                            } else {
                                acD.fx.washFilter.frequency.setTargetAtTime(100, audioCtx.currentTime, 0.05);
                                acD.fx.washFilter.Q.setTargetAtTime(1, audioCtx.currentTime, 0.05);
                            }
                        }
                    }
                });
            }

            if (progress >= 1) {
                clearInterval(interval);
                stopDeck(currentDeckIdx);
                isAutoMixingRef.current = false;

                // ── Stage 4: Cleanup ──────────────────────────────────────────
                if (audioCtx) {
                    [0, 1].forEach(di => {
                        const acD = audioChain.decks[di];
                        acD.fx.echoGain.gain.setTargetAtTime(0, audioCtx.currentTime, getSmoothTime(1));
                        acD.fx.reverbGain.gain.setTargetAtTime(0, audioCtx.currentTime, getSmoothTime(1));
                        acD.fx.washFilter.frequency.setTargetAtTime(100, audioCtx.currentTime, getSmoothTime(1));
                        acD.fx.washFilter.Q.setTargetAtTime(1, audioCtx.currentTime, getSmoothTime());
                        const cleanCurve = new Float32Array(44100);
                        for (let i = 0; i < 44100; i++) {
                            const x = (i * 2) / 44100 - 1;
                            cleanCurve[i] = ((3) * x * 20 * (Math.PI / 180)) / (Math.PI);
                        }
                        acD.fx.crush.curve = cleanCurve;

                        // Reset Filter
                        const filter = audioChain.decks[di].fx.djFilter;
                        filter.type = 'allpass';
                        filter.frequency.setTargetAtTime(1000, audioCtx.currentTime, 0.1);
                        filter.Q.setTargetAtTime(1, audioCtx.currentTime, 0.1);

                        // If incoming deck, reset to 100%. If outgoing deck, kill volume to 0.
                        const targetVol = di === nextDeckIdx ? 1.0 : 0.0;

                        // Reset all input gains and stem faders
                        acD.stereoInput.gain.setTargetAtTime(1, audioCtx.currentTime, 0.1);
                        acD.stereoFader.gain.setTargetAtTime(targetVol, audioCtx.currentTime, 0.1);
                        acD.stemFader.gain.setTargetAtTime(targetVol, audioCtx.currentTime, 0.1);

                        acD.stemBus.gain.setTargetAtTime(1, audioCtx.currentTime, 0.1);
                        acD.msMatrix.midGain.gain.setTargetAtTime(0.5, audioCtx.currentTime, 0.1);
                        acD.msMatrix.sideGain.gain.setTargetAtTime(0.5, audioCtx.currentTime, 0.1);

                        ['vocals', 'drums', 'bass', 'other'].forEach(stem => {
                            acD.stemInputs[stem as keyof typeof acD.stemInputs].gain.setTargetAtTime(1, audioCtx.currentTime, 0.1);
                            if (di === nextDeckIdx) {
                                acD.strips[stem as keyof typeof acD.strips].gain.gain.setTargetAtTime(1, audioCtx.currentTime, 0.1);
                            }
                        });
                    });

                    // Reset Master Bus just in case Headroom management dipped it hard
                    audioChain.masterBus.gain.setTargetAtTime(1, audioCtx.currentTime, 0.1);

                    setMasterBpm(targetBpm);
                    setMasterState(prev => ({ ...prev, crossfader: currentDeckIdx === 0 ? 1 : -1 }));
                }

                setDeckFX([
                    { ECHO: { active: false, value: 0.5 }, REVERB: { active: false, value: 0.4 }, CRUSH: { active: false, value: 0 }, WASH: { active: false, value: 0 } },
                    { ECHO: { active: false, value: 0.5 }, REVERB: { active: false, value: 0.4 }, CRUSH: { active: false, value: 0 }, WASH: { active: false, value: 0 } }
                ]);
                telemetry.success(`LIVE: ${nextTrack!.name}`);
            }
        }, tickTime);
    };

    const handleCrossfaderChange = (val: number) => {
        crossfaderRef.current = val;
        setMasterState(prev => ({ ...prev, crossfader: val }));

        // MIXING ENGINEER UPGRADE: Constant Power Fading
        const gainA = Math.cos((val + 1) * 0.25 * Math.PI);
        const gainB = Math.sin((val + 1) * 0.25 * Math.PI);
        audioChain.decks[0].stereoFader.gain.setTargetAtTime(gainA, audioCtx.currentTime, 0.05);
        audioChain.decks[1].stereoFader.gain.setTargetAtTime(gainB, audioCtx.currentTime, 0.05);

        // MIXING ENGINEER UPGRADE: Bass Swap EQ Ducking
        // As we fade to Deck B, pull Deck A's bass aggressively to avoid phase cancellation
        if (audioChain.decks[0].fx.isolator?.low) {
            const bassA = val > 0 ? Math.max(0, 1 - val * 2) : 1;
            const bassB = val < 0 ? Math.max(0, 1 + val * 2) : 1;
            audioChain.decks[0].fx.isolator.low.gain.setTargetAtTime(bassA, audioCtx.currentTime, 0.05);
            audioChain.decks[1].fx.isolator.low.gain.setTargetAtTime(bassB, audioCtx.currentTime, 0.05);
        }
    };

    const loadTrackToDeck = async (trackId: string, deckIdx: number, isAiTriggered = false, forcedStartTime?: number) => {
        if (!audioCtx) return;
        const track = tracksRef.current.find(t => t.id === trackId);
        if (!track || !track.buffer) return;

        const trackBpm = track.detectedBpm || DEFAULT_BPM;

        // If it's the first track playing OR we don't have Sync enabled on this deck, 
        // we might not want to instantly rate-stretch. But AI triggers should sync.
        const shouldSync = deckSync[deckIdx] || isAiTriggered;
        const rate = (isAiTriggered && !isPlaying) ? 1 : (shouldSync ? (masterBpm / trackBpm) : 1);
        const startTimeOffset = forcedStartTime !== undefined ? forcedStartTime : (track.structure?.musicStartTime || 0);

        // 1. Play Buffer A (Stereo) via AudioChain
        audioChain.loadTrack(deckIdx, track.buffer, startTimeOffset);

        // 2. Set Metadata
        decksRef.current[deckIdx] = {
            ...decksRef.current[deckIdx],
            trackId: track.id,
            startTime: audioCtx.currentTime - (startTimeOffset / rate), // Anchor time
            pauseTime: null,
            playbackRate: rate,
            isPlaying: true,
            duration: track.buffer.duration,
            active: true
        };

        // 3. Set BPM / Clock if First Track
        if (!isPlaying || (!isAutoMixingRef.current && !isAiTriggered && deckSync.filter(x => x).length < 2)) {
            setMasterBpm(trackBpm);
            clock.setBpm(trackBpm);
        }

        // 4. Stems Logic (Dual-Buffer)
        if (track.stemsReady && track.stemBuffers) {
            telemetry.info(`STEMS READY: ${track.id}`);
        }

        // Update UI State
        setActiveDeckIndex(deckIdx);
        setIsPlaying(true);
        playHistoryRef.current = [...playHistoryRef.current.slice(-9), track];

        if (!isAiTriggered) {
            telemetry.success(`LOADED: ${track.name}`);
        }
    };

    const stopDeck = (deckIdx: number) => {
        // Stop audio via Chain
        audioChain.stopDeck(deckIdx);

        // Update Metadata
        const deck = decksRef.current[deckIdx];
        deck.active = false;
        deck.isPlaying = false;

        // Update UI logic
        // We generally rely on deck.active so this is fine.
    };

    const seekDeck = (deckIdx: number, targetTime: number) => {
        if (!audioCtx) return;
        const deck = decksRef.current[deckIdx];
        const track = tracksRef.current.find(t => t.id === deck.trackId);
        if (!track || !track.buffer) return;

        // Clamp to valid range
        const clampedTime = Math.max(0, Math.min(track.buffer.duration - 0.1, targetTime));

        // Re-load track at the new offset (AudioSignalChain handles stop + recreate)
        audioChain.loadTrack(deckIdx, track.buffer, clampedTime);

        // Update deck metadata to reflect the new anchor time
        deck.startTime = audioCtx.currentTime - (clampedTime / deck.playbackRate);
        deck.pauseTime = null;
        deck.isPlaying = true;
        deck.active = true;

        // Re-apply playback rate for sync
        audioChain.setPlaybackRate(deckIdx, deck.playbackRate);

        telemetry.info(`SEEK: DECK_0${deckIdx + 1} → ${Math.floor(clampedTime / 60)}:${Math.floor(clampedTime % 60).toString().padStart(2, '0')}`);
    };

    const workerRef = useRef<Worker | null>(null);

    useEffect(() => {
        workerRef.current = new Worker(new URL('./services/analysis.worker.ts', import.meta.url), { type: 'module' });
        return () => workerRef.current?.terminate();
    }, []);

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || !audioCtx || !workerRef.current) return;
        const files = Array.from(e.target.files) as File[];

        if (files.length === 4) {
            telemetry.warn("STRESS MODE ACTIVATED: 4-TRACK SIMULTANEOUS INGESTION...");
        } else {
            telemetry.info("VIBRATION DETECTED: UPLOADING TO VAULT...");
        }

        for (const file of files) {
            try {
                // Read ArrayBuffer synchronously
                let arrayBuffer: ArrayBuffer;
                if (file.arrayBuffer) {
                    arrayBuffer = await file.arrayBuffer();
                } else if ((file as unknown as any).buffer) {
                    arrayBuffer = (file as unknown as any).buffer; // Chaos Test mock shortcut
                } else {
                    telemetry.error("INVALID FILE STRUCTURE DETECTED.");
                    continue;
                }

                // Generate ID
                const localId = `pending_${Math.random().toString(36).substr(2, 9)}`;

                // Decode audio data using browser's native multithreaded decoder (returns Promise, does not heavily block main thread)
                const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

                // Extract channels to pass to worker
                const channels: ArrayBuffer[] = [];
                for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
                    channels.push(audioBuffer.getChannelData(i).buffer);
                }

                // Message listener specifically for this file's decode response
                const handleMessage = (msgEvent: MessageEvent) => {
                    const data = msgEvent.data;
                    if (data.id === localId) {
                        workerRef.current?.removeEventListener('message', handleMessage);

                        if (data.status === 'success') {
                            // Rebuild AudioBuffer from returned arrays
                            const reconstructedBuffer = audioCtx.createBuffer(
                                audioBuffer.numberOfChannels,
                                data.length,
                                audioBuffer.sampleRate
                            );
                            for (let i = 0; i < data.channels.length; i++) {
                                reconstructedBuffer.copyToChannel(new Float32Array(data.channels[i]), i);
                            }

                            const newTrack: Track = {
                                id: localId,
                                file,
                                name: file.name.toUpperCase().replace(/\.[^/.]+$/, ""),
                                duration: data.duration,
                                buffer: reconstructedBuffer,
                                status: 'queued',
                                bpm: data.bpm || DEFAULT_BPM,
                                stemsReady: false,
                                stemsSeparating: false,
                            };

                            setTracks(prev => [...prev, newTrack]);

                            // Delegate to AnalysisQueue
                            analysisQueue.addJob({
                                trackId: localId,
                                file,
                                buffer: reconstructedBuffer,
                                status: 'queued',
                                onStatusUpdate: async (id, status, jobData) => {
                                    if (status === 'analyzing' && jobData) {
                                        setTracks(prev => prev.map(t => t.id === id ? { ...t, ...jobData } : t));
                                    }
                                    else if (status === 'stems_lite' && jobData) {
                                        const serverId = jobData.stemUrls.track_id;
                                        const liteBuffers = await fetchStemBuffers(jobData.stemUrls, audioCtx);

                                        setTracks(prev => prev.map(t => t.id === id ? {
                                            ...t,
                                            id: serverId,
                                            stemBuffers: liteBuffers,
                                            stemsReady: true,
                                            status: 'ready' as any
                                        } : t));

                                        if (isTakeoverMode && tracksRef.current.length === 1 && !isPlaying) {
                                            setTimeout(() => loadTrackToDeck(serverId, 0, true), 100);
                                        }

                                        if (jobData.stemUrls.hq_available) {
                                            pollForHQStems(serverId, audioCtx, (hqBuffers) => {
                                                setTracks(prev => prev.map(t => t.id === serverId ? { ...t, stemBuffers: hqBuffers } : t));
                                                telemetry.success(`NEURAL_UPGRADE: HQ STEMS DOWNLOADED`);
                                            });
                                        }

                                        fetch(`http://localhost:8000/analyze/rgb/${serverId}`)
                                            .then(r => r.ok ? r.json() : Promise.reject('RGB analysis failed'))
                                            .then(rgbData => {
                                                setTrackWaveforms(prev => ({ ...prev, [serverId]: rgbData }));
                                                telemetry.success(`RGB: SPECTRAL READY (ENERGY: ${rgbData.energy_score})`);
                                            });
                                    }
                                    else if (status === 'embeddings' && jobData) {
                                        setTracks(prev => prev.map(t => {
                                            if (t.id === id || (jobData.serverId && t.id === jobData.serverId)) {
                                                const defaultVec = Array.from({ length: 16 }, () => Math.random() - 0.5);
                                                let vEmb = jobData.embeddings?.vibeEmbedding || defaultVec;
                                                let aEmb = jobData.embeddings?.artistEmbedding || defaultVec;

                                                const updated = {
                                                    ...t,
                                                    embedding: vEmb,
                                                    artistEmbedding: aEmb
                                                };
                                                registerTrackWithSimilarityServer(updated);
                                                return updated;
                                            }
                                            return t;
                                        }));
                                    }
                                    else if (status === 'failed') {
                                        telemetry.error(`VAULT BREACH: ${id} ANALYSIS FAILED`);
                                        triggerGlitch(1.0);
                                        setTracks(prev => prev.map(t => t.id === id ? { ...t, status: 'error' as any } : t));
                                    }
                                }
                            });
                        } else {
                            console.error('Core Analysis Breach:', data.error);
                            telemetry.error(`STABILITY LOST: CORRUPT UPLOAD`);
                            triggerGlitch(1.0);
                        }
                    }
                };

                workerRef.current.addEventListener('message', handleMessage);

                // Transfer memory block of decoded float data for heavy analysis tasks
                workerRef.current.postMessage({
                    id: localId,
                    channels: channels,
                    length: audioBuffer.length,
                    duration: audioBuffer.duration
                }, channels);

            } catch (err) {
                console.error('Core Analysis Breach:', err);
                telemetry.error(`STABILITY LOST: CORRUPT UPLOAD`);
                triggerGlitch(1.0);
            }
        }
    };



    const handleUpdateEQ = (deckIdx: number, range: 'low' | 'mid' | 'high', val: number) => {
        setDeckEQ(p => {
            const next = [...p];
            next[deckIdx] = { ...next[deckIdx], [range]: val };
            return next;
        });

        // Use a standard DJ curve (0..1 maps to 0..unity, or more if needed)
        // For professional isolators, we want "Unity" at 0.5 or 1.0? 
        // Let's use 0..1 where 1 is unity (0dB) and 0 is -inf (kill).
        const gainNode = audioChain.decks[deckIdx].fx.isolator[range];
        gainNode.gain.setTargetAtTime(val, audioChain.ctx.currentTime, 0.05);
    };

    const handleUpdateStem = (deckIdx: number, stemId: string) => {
        if (!audioCtx) return;
        setDeckStems(prev => {
            const next = [...prev];
            const current = { ...next[deckIdx] };
            const key = stemId.toLowerCase() as keyof typeof current;
            current[key] = current[key] === 1 ? 0 : 1;
            next[deckIdx] = current;

            // Apply to Audio Chain Strips (Quantized)
            const strips = audioChain.decks[deckIdx].strips;

            // Map UI stem names to Strip parameters
            let targetParam: AudioParam | null = null;
            let targetVal = current[key];

            if (key === 'vocal') targetParam = strips.vocals.gain.gain;
            if (key === 'bass') targetParam = strips.bass.gain.gain;
            if (key === 'instru') targetParam = strips.other.gain.gain;
            if (key === 'kick' || key === 'hihat') {
                targetParam = strips.drums.gain.gain;
                // Logic: If either kick or hihat is active, the drum stem is active
                targetVal = (current.kick + current.hihat) > 0 ? 1 : 0;
            }

            if (targetParam) {
                quantizer.scheduleParameterChange(audioCtx, targetParam, targetVal, 0.010);
            }

            return next;
        });
    };

    const handleStemModeToggle = (deckIdx: number) => {
        const deck = decksRef.current[deckIdx];
        const isNowStemMode = !deck.usingStemMode;

        deck.usingStemMode = isNowStemMode;
        audioChain.decks[deckIdx].usingStemMode = isNowStemMode;

        if (isNowStemMode && deck.active) {
            const track = tracksRef.current.find(t => t.id === deck.trackId);
            if (track && track.stemsReady && track.stemBuffers) {
                telemetry.info(`STEM_ENGAGE: MIGRATING DECK ${deckIdx === 0 ? 'A' : 'B'} TO NEURAL ROUTING`);
                const offset = audioCtx ? (audioCtx.currentTime - deck.startTime) : 0;
                audioChain.hotSwapStems(deckIdx, track.stemBuffers, offset, deck.playbackRate);
            } else {
                telemetry.warn("STEM_ENGAGE: STEMS NOT READY YET");
            }
        } else if (!isNowStemMode) {
            telemetry.info(`STEM_DISENGAGE: DECK ${deckIdx === 0 ? 'A' : 'B'} REVERTING TO STEREO`);
            // @ts-ignore
            audioChain.revertToStereo(deckIdx);
        }

        forceRender({});
    };

    // ── MASHUP PROTOCOL ──────────────────────────────────────────────────────
    // Layers Vocals from the Active Deck + Drums/Bass from the Other Deck
    const executeMashup = () => {
        if (!audioCtx) return;
        const deckARef = decksRef.current[0];
        const deckBRef = decksRef.current[1];

        // Both decks must be playing with stems ready
        const trackA = tracksRef.current.find(t => t.id === deckARef.trackId);
        const trackB = tracksRef.current.find(t => t.id === deckBRef.trackId);

        if (!trackA?.stemsReady || !trackA?.stemBuffers || !trackB?.stemsReady || !trackB?.stemBuffers) {
            telemetry.warn('MASHUP: STEMS NOT READY ON BOTH DECKS. WAITING...');
            return;
        }

        if (!deckARef.active || !deckBRef.active) {
            telemetry.warn('MASHUP: BOTH DECKS MUST BE ACTIVE. LOAD TRACKS FIRST.');
            return;
        }

        telemetry.ai('MASHUP PROTOCOL: ENGAGING CROSS-STEM LAYERING...');

        // Ensure both decks are in stem mode
        [0, 1].forEach(idx => {
            const deck = decksRef.current[idx];
            const track = tracksRef.current.find(t => t.id === deck.trackId);
            if (!deck.usingStemMode && track?.stemBuffers) {
                deck.usingStemMode = true;
                audioChain.decks[idx].usingStemMode = true;
                const offset = (audioCtx.currentTime - deck.startTime) * deck.playbackRate;
                audioChain.hotSwapStems(idx, track.stemBuffers, offset, deck.playbackRate);
            }
        });

        const now = audioCtx.currentTime;
        const smooth = 0.2;

        // Deck A: VOCALS + OTHER full, DRUMS + BASS muted
        audioChain.decks[0].strips.vocals.gain.gain.setTargetAtTime(1.0, now, smooth);
        audioChain.decks[0].strips.other?.gain.gain.setTargetAtTime(0.7, now, smooth);
        audioChain.decks[0].strips.drums.gain.gain.setTargetAtTime(0.0, now, smooth);
        audioChain.decks[0].strips.bass.gain.gain.setTargetAtTime(0.0, now, smooth);

        // Deck B: DRUMS + BASS full, VOCALS + OTHER muted
        audioChain.decks[1].strips.drums.gain.gain.setTargetAtTime(1.0, now, smooth);
        audioChain.decks[1].strips.bass.gain.gain.setTargetAtTime(1.0, now, smooth);
        audioChain.decks[1].strips.vocals.gain.gain.setTargetAtTime(0.0, now, smooth);
        audioChain.decks[1].strips.other?.gain.gain.setTargetAtTime(0.3, now, smooth);

        // Ensure both faders are open
        audioChain.decks[0].stemFader.gain.setTargetAtTime(1, now, 0.01);
        audioChain.decks[1].stemFader.gain.setTargetAtTime(1, now, 0.01);

        setIsMashupActive(true);
        telemetry.success('MASHUP: VOCALS(A) + DRUMS(B) LAYERED — LIVE');
    };

    const disengageMashup = () => {
        if (!audioCtx) return;
        const now = audioCtx.currentTime;
        const smooth = 0.3;

        // Restore all stem gains to 1.0
        [0, 1].forEach(idx => {
            ['vocals', 'drums', 'bass', 'other'].forEach(stem => {
                const strip = audioChain.decks[idx].strips[stem as keyof typeof audioChain.decks[0]['strips']];
                if (strip) {
                    strip.gain.gain.setTargetAtTime(1.0, now, smooth);
                }
            });
        });

        setIsMashupActive(false);
        telemetry.info('MASHUP: DISENGAGED — ALL STEMS RESTORED');
    };

    const handleUpdateFX = (deckIdx: number, type: FXType, updates: Partial<{ active: boolean; value: number }>) => {
        if (!audioCtx) return;
        setDeckFX(prev => {
            const next = [...prev];
            next[deckIdx] = { ...next[deckIdx], [type]: { ...next[deckIdx][type], ...updates } };
            const fxState = next[deckIdx][type];
            // Call Audio Chain
            audioChain.updateFX(deckIdx, type, fxState);
            return next;
        });
    };

    const handleStemSwap = async (deckIdx: number, stemId: string) => {
        if (!audioCtx) return;
        const targetDeck = decksRef.current[deckIdx];
        if (!targetDeck) return;

        // Compatible Candidates: Not self, and has stems
        const candidates = tracksRef.current.filter(t =>
            t.id !== targetDeck.trackId &&
            t.stemsReady &&
            t.stemBuffers
        );

        if (candidates.length === 0) {
            telemetry.warn("MASHUP: NO COMPATIBLE STEMS FOUND");
            return;
        }

        const randomTrack = candidates[Math.floor(Math.random() * candidates.length)];
        const stems = randomTrack.stemBuffers!;

        const stemKey = stemId.toLowerCase() as 'vocals' | 'drums' | 'bass' | 'other';
        const newBuffer = stems[stemKey];

        if (!newBuffer) {
            telemetry.warn(`MASHUP: ${stemId.toUpperCase()} STEM MISSING`);
            return;
        }

        // Pitch Shift to match Master BPM
        const trackBpm = randomTrack.detectedBpm || randomTrack.bpm || 128;
        const pitchRate = deckSync[deckIdx] ? (masterBpm / trackBpm) : 1;

        await audioChain.swapStem(
            deckIdx,
            stemKey,
            newBuffer,
            pitchRate,
            deckPlayback[deckIdx].current
        );

        telemetry.info(`MASHUP: SWAPPED ${stemId.toUpperCase()} w/ ${randomTrack.name.toUpperCase()}`);
    };

    const [, forceRender] = useState({});


    const handleStart = async () => {
        const ctx = audioChain.ctx;
        if (ctx.state === 'suspended') await ctx.resume();

        // Start Clock
        clock.isPlaying = true;
        clock.lastTime = performance.now();

        setAudioCtx(ctx);
        masterRef.current = { analyser: audioChain.masterAnalyser };
        setHasStarted(true);
        telemetry.info("SYSTEM: AUDIO ENGINE STARTED");
    };

    if (!hasStarted) {
        return <WelcomeScreen onEnter={handleStart} />;
    }

    return (
        <div className="w-full h-screen relative flex flex-col p-2 gap-2 overflow-hidden select-none bg-dot-grid">
            {/* HEADER BAR (3-COLUMN SYSTEM) */}
            <header className="h-14 grid grid-cols-[1fr_auto_1fr] items-center px-4 glass-panel bg-panel-bg-solid z-50 overflow-hidden">
                {/* Left Section: Brand & Phase */}
                <div className="flex items-center gap-4 min-w-0">
                    <div className="flex flex-col shrink-0">
                        <span className="font-dot text-sm text-accent tracking-widest leading-none">AI_BOILER_ROOM // NEURAL_DJ</span>
                        <span className="font-dot text-[8px] text-dim tracking-[0.2em] mt-1 uppercase">PHASE: {isTakeoverMode ? 'AUTONOMOUS' : 'MANUAL'}</span>
                    </div>
                </div>

                {/* Center Section: AI Telemetry & BPM Hub */}
                <div className="flex items-center gap-4 px-6 bg-main/5 rounded-sm border-x border-main/10">
                    {/* Master BPM */}
                    <div className="flex flex-col items-center shrink-0">
                        <span className="text-[7px] font-dot text-dim uppercase tracking-tighter">BPM</span>
                        <div className="flex items-center gap-1">
                            <span className="text-xl font-bold font-mono text-accent digital-readout">{Math.round(masterBpm)}</span>
                        </div>
                    </div>

                    <div className="h-8 w-[1px] bg-main opacity-10"></div>

                    {/* Neural Toggle */}
                    <button onClick={() => setIsTakeoverMode(!isTakeoverMode)} className={`flex items-center gap-2 px-3 py-1.5 border rounded-sm btn-hardware shrink-0 ${isTakeoverMode ? 'border-accent text-accent' : 'text-dim'}`} aria-label="Toggle Neural DJ Mode">
                        <BrainCircuit size={14} className={isTakeoverMode ? 'animate-pulse' : ''} />
                        <div className={`led ${isTakeoverMode ? 'active' : ''}`} />
                    </button>

                    <div className="h-8 w-[1px] bg-main opacity-10"></div>

                    {/* Ndot Musical Meters */}
                    <div className="flex items-center gap-4 shrink-0 whitespace-nowrap">
                        <div className="flex flex-col w-[80px]">
                            <span className="text-[6px] font-dot text-dim uppercase tracking-tighter">ENERGY</span>
                            <span className="text-[10px] font-mono text-accent">E [ {":".repeat(Math.ceil(energyScore / 3))}... ]</span>
                        </div>
                        <div className="flex flex-col w-[80px]">
                            <span className="text-[6px] font-dot text-dim uppercase tracking-tighter">LUFS</span>
                            <span className={`text-[10px] font-mono ${lufsReading.isClipping ? 'text-red-500 animate-pulse font-bold' : 'text-accent'}`}>
                                {isFinite(lufsReading.momentary) ? lufsReading.momentary.toFixed(1) : '--'}
                            </span>
                        </div>
                        <div className="flex flex-col w-[80px]">
                            <span className="text-[6px] font-dot text-dim uppercase tracking-tighter">PEAK</span>
                            <span className={`text-[10px] font-mono ${lufsReading.truePeak > -1 ? 'text-red-500 font-bold' : lufsReading.truePeak > -3 ? 'text-yellow-500' : 'text-accent'}`}>
                                {isFinite(lufsReading.truePeak) ? `${lufsReading.truePeak.toFixed(1)}dB` : '--'}
                            </span>
                        </div>
                        <div className="flex flex-col w-[80px]">
                            <span className="text-[6px] font-dot text-dim uppercase tracking-tighter">PHRASE</span>
                            <span className="text-[10px] font-mono text-accent">P [ {phraseCountdown.toString().padStart(2, '0')} ]</span>
                        </div>
                    </div>

                    <div className="h-8 w-[1px] bg-main opacity-10"></div>

                    {/* Transition Prediction */}
                    <div className="flex items-center gap-3 shrink-0 whitespace-nowrap">
                        <div className="flex flex-col w-[80px]">
                            <span className="text-[6px] font-dot text-dim uppercase tracking-tighter">TRANSITION</span>
                            <span className="text-[10px] font-mono text-accent">
                                PROB: {lookaheadRef.current ? "|||||" : "....."}
                            </span>
                        </div>
                        <div className="flex flex-col">
                            <span className="text-[6px] font-dot text-dim uppercase tracking-tighter">STYLE</span>
                            <span className="text-[10px] font-mono text-accent">
                                [ {lookaheadRef.current?.style || 'R E A D I N G'} ]
                            </span>
                        </div>
                    </div>
                </div>

                {/* Right Section: System Controls */}
                <div className="flex justify-end items-center gap-4">
                    {/* Creativity Control (Moved to System area) */}
                    <div className="flex flex-col items-center">
                        <span className="text-[7px] font-dot text-dim uppercase mb-0.5 tracking-tighter">CREATIVITY</span>
                        <div className="flex items-center gap-2 group">
                            <span className={`text-[8px] font-dot ${creativity < 0.3 ? 'text-accent' : 'text-dim'}`}>TRACK</span>
                            <input
                                type="range" min="0" max="1" step="0.01"
                                value={creativity}
                                onChange={(e) => setCreativity(parseFloat(e.target.value))}
                                className="w-16 accent-accent cursor-pointer h-1"
                                aria-label="Creativity Level"
                            />
                            <span className={`text-[8px] font-dot ${creativity > 0.7 ? 'text-accent' : 'text-dim'}`}>ARTIST</span>
                        </div>
                    </div>

                    <div className="h-8 w-[1px] bg-main opacity-10"></div>

                    <div className="h-8 w-[1px] bg-main opacity-10"></div>

                    <button
                        onClick={() => {
                            setSkipScheduled(true);
                            telemetry.warn("MANUAL OVERRIDE: INITIATING IMMEDIATE TRANSITION");
                        }}
                        className={`p-2 transition-colors ${skipScheduled ? 'text-accent animate-pulse shadow-[0_0_10px_rgba(255,59,48,0.3)]' : 'text-dim hover:text-accent'}`}
                        aria-label="Skip Track"
                        title="Force Immediate Transition"
                    >
                        <FastForward size={18} />
                    </button>

                    <div className="h-8 w-[1px] bg-main opacity-10"></div>

                    <div className="flex gap-2">
                        <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} className="p-2 text-dim hover:text-accent">
                            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
                        </button>
                        <button onClick={() => window.location.reload()} className="p-2 text-dim hover:text-red-500" aria-label="Reload Application">
                            <Power size={18} />
                        </button>
                    </div>
                </div>
            </header>

            {/* WORKSPACE GRID */}
            < div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-2 min-h-0 relative" >
                {/* DECK A */}
                < ErrorBoundary fallbackName="DECK_ALPHA" >
                    <motion.div
                        initial={{ opacity: 0, x: -50 }}
                        animate={{ opacity: activeDeckIndex === 0 ? 1 : 0.6, x: 0, scale: activeDeckIndex === 0 ? 1 : 0.98 }}
                        transition={{ type: "spring", stiffness: 100, damping: 20 }}
                        className={`md:col-span-3 flex flex-col glass-panel overflow-hidden transition-all duration-500 
                            ${activeDeckIndex === 0 ? 'ring-1 ring-accent shadow-[0_0_20px_rgba(255,59,48,0.2)] bg-accent/5' : 'opacity-60 grayscale-[0.5]'}
                            h-full relative`}
                    >
                        {decksRef.current[0]?.trackId && trackWaveforms[decksRef.current[0].trackId] && (
                            <div className="absolute inset-0 z-0 opacity-40 pointer-events-none mix-blend-screen bg-black/80">
                                <RGBWaveform
                                    rgbData={trackWaveforms[decksRef.current[0].trackId].data}
                                    zoom={1.5}
                                    progress={deckPlayback[0].duration > 0 ? (deckPlayback[0].current / deckPlayback[0].duration) : 0}
                                    collision={activeDeckIndex === 0 ? spectralCollision : 0}
                                    glitchIntensity={glitchIntensity}
                                />
                            </div>
                        )}
                        <div className="relative z-10 flex-1 flex flex-col h-full min-h-0">
                            <DeckControls
                                deckId="A"
                                stems={deckStems[0]}
                                filterValue={deckFilters[0]}
                                fxState={deckFX[0]}
                                loopState={deckLoops[0]}
                                onUpdateFilter={(val) => {
                                    if (!audioCtx) return;
                                    setDeckFilters(p => { const n = [...p]; n[0] = val; return n; });
                                    const filter = audioChain.decks[0].fx.djFilter;
                                    const smooth = getSmoothTime(0.2);

                                    if (val > 0.485 && val < 0.515) {
                                        filter.type = 'allpass';
                                        filter.frequency.setTargetAtTime(1000, audioCtx.currentTime, smooth);
                                    } else if (val <= 0.485) {
                                        filter.type = 'lowpass';
                                        const norm = val / 0.485;
                                        const freq = 20 * Math.pow(1000, norm);
                                        filter.frequency.setTargetAtTime(freq, audioCtx.currentTime, smooth);
                                    } else {
                                        filter.type = 'highpass';
                                        const norm = (val - 0.515) / 0.485;
                                        const freq = 20 * Math.pow(1000, norm);
                                        filter.frequency.setTargetAtTime(freq, audioCtx.currentTime, smooth);
                                    }
                                }}
                                onUpdateFX={(type, updates) => handleUpdateFX(0, type, updates)}
                                onLoopToggle={() => {
                                    if (!audioCtx) return;
                                    setDeckLoops(p => { const n = [...p]; n[0].active = !n[0].active; return n; });
                                    const deck = decksRef.current[0];
                                    if (audioChain.decks[0].bufferASource) {
                                        audioChain.decks[0].bufferASource.loop = !deckLoops[0].active;
                                        if (audioChain.decks[0].bufferASource.loop) {
                                            audioChain.decks[0].bufferASource.loopStart = (audioCtx.currentTime - deck.startTime) * deck.playbackRate;
                                            audioChain.decks[0].bufferASource.loopEnd = audioChain.decks[0].bufferASource.loopStart + (60 / masterBpm * deckLoops[0].length);
                                        }
                                    }
                                }}
                                onLoopLengthChange={(l) => setDeckLoops(p => { const n = [...p]; n[0].length = l; return n; })}
                                isPlaying={deckPlayback[0].isPlaying}
                                track={tracks.find(t => t.id === decksRef.current[0]?.trackId)}
                                duration={deckPlayback[0].duration}
                                currentTime={deckPlayback[0].current}
                                analyser={decksRef.current[0]?.analyser}
                                onStemToggle={(s) => handleUpdateStem(0, s)}
                                onStemSwap={(s) => handleStemSwap(0, s)}
                                onStop={() => stopDeck(0)}
                                usingStemMode={decksRef.current[0]?.usingStemMode || false}
                                onStemModeToggle={() => handleStemModeToggle(0)}
                                eq={deckEQ[0]}
                                onUpdateEQ={(range, val) => handleUpdateEQ(0, range, val)}
                                syncEnabled={deckSync[0]}
                                onSyncToggle={() => {
                                    setDeckSync(prev => {
                                        const next = [...prev];
                                        next[0] = !next[0];
                                        return next;
                                    });
                                }}
                                onNudge={(amt) => {
                                    if (!audioCtx) return;
                                    const d = decksRef.current[0];
                                    audioChain.setPlaybackRate(0, d.playbackRate + amt * 0.05);
                                }}
                                stemAnalysers={audioChain.decks[0].stemAnalysers}
                                onSeek={(t) => seekDeck(0, t)}
                            />
                        </div>
                    </motion.div>
                </ErrorBoundary >

                {/* CENTER HUB */}
                < ErrorBoundary fallbackName="CORE_PROCESSOR" >
                    <div className={`md:col-span-6 flex flex-col gap-2 min-h-0 h-full`}>
                        <div className="flex-1 glass-panel p-2 flex flex-col gap-2 relative bg-panel-bg-solid/40 min-h-0">
                            <div className="flex justify-between items-center px-1 border-b border-main/20 pb-1">
                                <div className="flex items-center gap-2">
                                    <Activity size={10} className="text-accent" />
                                    <span className="font-dot text-[9px] tracking-[0.4em] text-dim uppercase">Live_Signal_Map</span>
                                </div>
                                {timeUntilTransition !== null && timeUntilTransition < 1000 ? (
                                    <div className="flex items-center gap-2 bg-accent/10 px-2 py-0.5 rounded-sm border border-accent/20">
                                        <span className="font-dot text-[8px] text-accent tracking-widest">NEXT_PHASE:</span>
                                        <span className="font-mono text-xs font-bold text-accent digital-readout">{timeUntilTransition > 0 ? timeUntilTransition : 'NOW'}S</span>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-2 px-2 py-0.5 opacity-50">
                                        <span className="font-dot text-[8px] text-dim tracking-widest">NEXT_PHASE:</span>
                                        <span className="font-mono text-xs font-bold text-dim digital-readout">--S</span>
                                    </div>
                                )}
                            </div>

                            {/* Live Visualizer Rendered Here */}
                            <div className="flex-1 relative overflow-hidden mt-1 rounded-sm border border-main/10 bg-black/50">
                                <Visualizer
                                    analyser={decksRef.current[activeDeckIndex]?.analyser || null}
                                    isActive={deckPlayback[activeDeckIndex]?.isPlaying || false}
                                    mode="spectrum"
                                />
                            </div>
                        </div>
                        <div className="flex-[2] flex flex-col glass-panel bg-panel-bg-solid/30 p-2 overflow-hidden border border-main/20 min-h-0">
                            <div className="flex items-center gap-2 mb-1 opacity-50 shrink-0"><ShieldAlert size={10} /><span className="font-dot text-[8px] uppercase tracking-widest">Telemetry_Log</span></div>
                            <div className="flex-1 overflow-y-auto custom-scrollbar font-mono text-[9px] min-h-0">
                                {/* UX/UI DESIGNER UPGRADE: Slice logs to prevent Virtual DOM lag and maintain micro-fluidity */}
                                {logs.slice(-40).map(log => {
                                    let typeColor = 'text-main opacity-80';
                                    if (log.level === 'ai') typeColor = 'text-accent font-bold';
                                    if (log.level === 'warn') typeColor = 'text-yellow-500';
                                    if (log.level === 'success') typeColor = 'text-green-500 shadow-[0_0_10px_rgba(34,197,94,0.3)]';

                                    return (
                                        <div key={log.id} className={`mb-1 border-l-2 ${log.level === 'ai' ? 'border-accent' : 'border-main/20'} pl-2 leading-relaxed transition-all duration-300 hover:bg-main/5`}>
                                            <span className="text-[7px] text-dim opacity-40 font-mono">[{log.timestamp.split(':')[2]}S] </span>
                                            <span className={`${typeColor} uppercase tracking-tight`}>{log.message}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                        {/* MASHUP Button + Crossfader */}
                        <div className="h-16 glass-panel flex flex-col justify-center px-8 bg-panel-bg-solid/60 relative group">
                            <div className="flex justify-between items-center text-[8px] font-dot text-dim uppercase mb-1">
                                <span className={masterState.crossfader < -0.9 ? 'text-accent' : ''}>DECK_01</span>
                                <button
                                    onClick={() => isMashupActive ? disengageMashup() : executeMashup()}
                                    className={`px-3 py-1 border font-dot text-[8px] uppercase tracking-widest rounded-sm transition-all duration-200 btn-hardware ${isMashupActive
                                        ? 'border-accent text-accent bg-accent/15 shadow-[0_0_12px_rgba(255,59,48,0.3)] animate-pulse'
                                        : 'border-main/30 text-dim hover:text-accent hover:border-accent/50'
                                        }`}
                                >
                                    {isMashupActive ? '[■ MASHUP]' : '[▶ MASHUP]'}
                                </button>
                                <span className={masterState.crossfader > 0.9 ? 'text-accent' : ''}>DECK_02</span>
                            </div>
                            <div
                                className="relative h-1 bg-main/5 flex items-center cursor-pointer"
                                onMouseDown={(e) => {
                                    const rect = e.currentTarget.getBoundingClientRect();
                                    const update = (clientX: number) => {
                                        const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
                                        handleCrossfaderChange((x * 2) - 1);
                                    };
                                    update(e.clientX);
                                    const onMouseMove = (moveEvent: MouseEvent) => update(moveEvent.clientX);
                                    const onMouseUp = () => {
                                        window.removeEventListener('mousemove', onMouseMove);
                                        window.removeEventListener('mouseup', onMouseUp);
                                    };
                                    window.addEventListener('mousemove', onMouseMove);
                                    window.addEventListener('mouseup', onMouseUp);
                                }}
                            >
                                {/* UI DESIGNER UPGRADE: Transform scaling on fader knob for technical rawness */}
                                <div className="absolute h-10 w-2.5 bg-accent shadow-[0_0_20px_rgba(255,59,48,0.4)] border border-panel z-10 transition-transform duration-75"
                                    style={{ left: `${(masterState.crossfader + 1) * 50}%`, marginLeft: '-5px', transform: `scaleX(${1 + Math.abs(masterState.crossfader) * 0.4})` }}></div>
                                <div className="absolute inset-0 flex justify-between px-0.5 opacity-20 pointer-events-none">
                                    {[...Array(21)].map((_, i) => <div key={i} className={`w-[1px] h-full ${i === 10 ? 'bg-accent opacity-100' : 'bg-main'}`} />)}
                                </div>
                            </div>
                        </div>
                    </div>
                </ErrorBoundary>
                {/* DECK B */}
                < ErrorBoundary fallbackName="DECK_BETA" >
                    <motion.div
                        initial={{ opacity: 0, x: 50 }}
                        animate={{ opacity: activeDeckIndex === 1 ? 1 : 0.6, x: 0, scale: activeDeckIndex === 1 ? 1 : 0.98 }}
                        transition={{ type: "spring", stiffness: 100, damping: 20 }}
                        className={`md:col-span-3 flex flex-col glass-panel overflow-hidden transition-all duration-500 
                            ${activeDeckIndex === 1 ? 'ring-1 ring-accent shadow-[0_0_20px_rgba(255,59,48,0.2)] bg-accent/5' : 'opacity-60 grayscale-[0.5]'}
                            h-full relative`}
                    >
                        {decksRef.current[1]?.trackId && trackWaveforms[decksRef.current[1].trackId] && (
                            <div className="absolute inset-0 z-0 opacity-40 pointer-events-none mix-blend-screen bg-black/80">
                                <RGBWaveform
                                    rgbData={trackWaveforms[decksRef.current[1].trackId].data}
                                    zoom={1.5}
                                    progress={deckPlayback[1].duration > 0 ? (deckPlayback[1].current / deckPlayback[1].duration) : 0}
                                    collision={activeDeckIndex === 1 ? spectralCollision : 0}
                                    glitchIntensity={glitchIntensity}
                                />
                            </div>
                        )}
                        <div className="relative z-10 flex-1 flex flex-col h-full min-h-0">
                            <DeckControls
                                deckId="B"
                                stems={deckStems[1]}
                                filterValue={deckFilters[1]}
                                fxState={deckFX[1]}
                                loopState={deckLoops[1]}
                                onUpdateFilter={(val) => {
                                    if (!audioCtx) return;
                                    setDeckFilters(p => { const n = [...p]; n[1] = val; return n; });
                                    const filter = audioChain.decks[1].fx.djFilter;
                                    const smooth = getSmoothTime(0.2);

                                    if (val > 0.485 && val < 0.515) {
                                        filter.type = 'allpass';
                                        filter.frequency.setTargetAtTime(1000, audioCtx.currentTime, smooth);
                                    } else if (val <= 0.485) {
                                        filter.type = 'lowpass';
                                        const norm = val / 0.485;
                                        const freq = 20 * Math.pow(1000, norm);
                                        filter.frequency.setTargetAtTime(freq, audioCtx.currentTime, smooth);
                                    } else {
                                        filter.type = 'highpass';
                                        const norm = (val - 0.515) / 0.485;
                                        const freq = 20 * Math.pow(1000, norm);
                                        filter.frequency.setTargetAtTime(freq, audioCtx.currentTime, smooth);
                                    }
                                }}
                                onUpdateFX={(type, updates) => handleUpdateFX(1, type, updates)}
                                onLoopToggle={() => {
                                    if (!audioCtx) return;
                                    setDeckLoops(p => { const n = [...p]; n[1].active = !n[1].active; return n; });
                                    const deck = decksRef.current[1];
                                    if (audioChain.decks[1].bufferASource) {
                                        audioChain.decks[1].bufferASource.loop = !deckLoops[1].active;
                                        if (audioChain.decks[1].bufferASource.loop) {
                                            audioChain.decks[1].bufferASource.loopStart = (audioCtx.currentTime - deck.startTime) * deck.playbackRate;
                                            audioChain.decks[1].bufferASource.loopEnd = audioChain.decks[1].bufferASource.loopStart + (60 / masterBpm * deckLoops[1].length);
                                        }
                                    }
                                }}
                                onLoopLengthChange={(l) => setDeckLoops(p => { const n = [...p]; n[1].length = l; return n; })}
                                isPlaying={deckPlayback[1].isPlaying}
                                track={tracks.find(t => t.id === decksRef.current[1]?.trackId)}
                                duration={deckPlayback[1].duration}
                                currentTime={deckPlayback[1].current}
                                analyser={decksRef.current[1]?.analyser}
                                onStemToggle={(s) => handleUpdateStem(1, s)}
                                onStemSwap={(s) => handleStemSwap(1, s)}
                                onStop={() => stopDeck(1)}
                                usingStemMode={decksRef.current[1]?.usingStemMode || false}
                                onStemModeToggle={() => handleStemModeToggle(1)}
                                eq={deckEQ[1]}
                                onUpdateEQ={(range, val) => handleUpdateEQ(1, range, val)}
                                syncEnabled={deckSync[1]}
                                onSyncToggle={() => {
                                    setDeckSync(prev => {
                                        const next = [...prev];
                                        next[1] = !next[1];
                                        return next;
                                    });
                                }}
                                onNudge={(amt) => {
                                    if (!audioCtx) return;
                                    const d = decksRef.current[1];
                                    audioChain.setPlaybackRate(1, d.playbackRate + amt * 0.05);
                                }}
                                stemAnalysers={audioChain.decks[1].stemAnalysers}
                                onSeek={(t) => seekDeck(1, t)}
                            />
                        </div>
                    </motion.div>
                </ErrorBoundary >
            </div >

            {/* FOOTER */}
            < div className="h-40 grid grid-cols-12 gap-2" >
                <div className="col-span-4 flex flex-col glass-panel overflow-hidden border border-main/20 bg-panel-bg-solid/30">
                    <div className="px-3 py-1.5 border-b border-main/20 flex justify-between items-center bg-panel-bg-solid/60">
                        <div className="flex items-center gap-2">
                            <LayoutList size={12} className="text-accent" />
                            <span className="font-dot text-[9px] text-dim uppercase tracking-widest">Signal_Vault</span>
                        </div>
                        <label className="cursor-pointer font-dot text-[8px] text-dim hover:text-accent border border-main/40 px-2 py-0.5 uppercase btn-hardware rounded-sm">
                            IMPORT
                            <input type="file" className="hidden" multiple accept="audio/*" onChange={handleUpload} />
                        </label>
                    </div>
                    <div className="flex-1 overflow-hidden">
                        <TrackList tracks={tracks} currentTrackId={decksRef.current[activeDeckIndex]?.trackId} onLoadA={(id) => loadTrackToDeck(id, 0)} onLoadB={(id) => loadTrackToDeck(id, 1)} onAddToQueue={(id) => setAutoMixQueue(p => [...p, id])} />
                    </div>
                </div>

                <div className="col-span-8 glass-panel p-2 flex flex-col overflow-hidden border border-main/20 bg-panel-bg-solid/30">
                    <div className="flex justify-between items-center border-b border-main/20 pb-1 mb-2">
                        <div className="flex items-center gap-2">
                            <ListOrdered size={12} className="text-dim" />
                            <span className="font-dot text-[10px] text-dim uppercase tracking-widest">Autonomous_Set_Sequence</span>
                        </div>
                        <div className="flex gap-2">

                            <button onClick={async () => {
                                if (!tracks.length) return;
                                setIsAiProcessing(true);
                                telemetry.ai("PLANNING OPTIMAL FLOW...");
                                const res = await sortSetFlow(tracks);
                                setAutoMixQueue(res);
                                setIsAiProcessing(false);
                                telemetry.success("SEQUENCE MAP READY");
                            }} disabled={isAiProcessing || tracks.length < 2} className="px-3 py-1 border border-main/40 font-dot text-[9px] text-accent hover:bg-accent/10 uppercase btn-hardware rounded-sm">
                                <Sparkles size={10} className="inline mr-1" /> OPTIMIZE
                            </button>
                        </div>
                    </div>
                    <div className="flex-1 flex gap-2 overflow-x-auto custom-scrollbar pb-1">
                        {autoMixQueue.length > 0 ? autoMixQueue.map((id, idx) => {
                            const t = tracks.find(track => track.id === id);
                            return (
                                <div key={`${id}-${idx}`} className="min-w-[130px] glass-panel p-2 flex flex-col justify-between border border-main/30 bg-panel-bg-solid/40 group hover:border-accent/40 transition-colors">
                                    <div className="flex justify-between items-start"><span className="font-dot text-[7px] text-dim opacity-60">SEQ_0{idx + 1}</span><button onClick={() => setAutoMixQueue(p => p.filter((_, i) => i !== idx))} className="text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"><Trash2 size={10} /></button></div>
                                    <span className="font-bold text-[9px] uppercase truncate text-accent mt-1 leading-none">{t?.name || 'SIGNAL'}</span>
                                    <span className="font-dot text-[8px] text-dim mt-auto opacity-40">{t?.detectedBpm || '--'} BPM</span>
                                </div>
                            );
                        }) : <div className="flex-1 flex items-center justify-center border border-dashed border-main/30 text-dim font-dot text-[10px] uppercase tracking-[0.5em] opacity-20">Sequence_Empty</div>}
                    </div>
                </div>
            </div >
        </div >
    );
};

export default App;
