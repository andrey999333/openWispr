package com.voicerewriter

import android.content.Context
import android.util.Log
import android.view.inputmethod.InputMethodManager
import com.whispercpp.whisper.WhisperContext
import com.whispercpp.whisper.WhisperCpuConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * On-device speech-to-text via whisper.cpp (vendored `:lib`). Lazily loads the
 * downloaded model into a cached WhisperContext (reused across dictations) and
 * transcribes 16 kHz mono float samples fully offline.
 *
 * Same role as [SttEngine] but local; the caller picks based on the STT provider.
 */
object LocalWhisperStt {

    private val loadLock = Mutex()
    @Volatile private var ctx: WhisperContext? = null
    @Volatile private var loadedId: String? = null

    /**
     * Transcribe [samples] (16 kHz mono, normalized -1..1). Suspends on heavy CPU work.
     * [biasPrompt] (optional) primes the recognizer toward the user's vocabulary.
     */
    suspend fun transcribe(context: Context, settings: Settings, samples: FloatArray, biasPrompt: String? = null): String {
        val id = settings.sttModel.ifBlank { WhisperModelManager.DEFAULT_MODEL }
        if (!WhisperModelManager.isReady(context, id)) {
            throw IllegalStateException("On-device model not downloaded. Open Settings → Voice → Download model.")
        }
        val seconds = samples.size / AudioRecorder.SAMPLE_RATE.toFloat()
        val t0 = System.nanoTime()
        val whisper = loadLock.withLock {
            if (ctx == null || loadedId != id) {
                ctx?.let { runCatching { it.release() } }
                ctx = withContext(Dispatchers.IO) {
                    WhisperContext.createContextFromFile(WhisperModelManager.modelFile(context, id).absolutePath)
                }
                loadedId = id
            }
            ctx!!
        }
        val tLoaded = System.nanoTime()
        // transcribeData runs on whisper's own single-thread dispatcher internally.
        val language = resolveLanguage(context, settings.sttLanguage)
        Log.i("LocalWhisperStt", "languageSetting=${settings.sttLanguage} resolvedLanguage=${language ?: "auto"}")
        val raw = whisper.transcribeData(
            samples,
            printTimestamp = false,
            prompt = biasPrompt?.ifBlank { null },
            language = language,
        )
        val tDone = System.nanoTime()
        Log.i(
            "LocalWhisperStt",
            "model=$id audio=${"%.1f".format(seconds)}s " +
                "load=${(tLoaded - t0) / 1_000_000}ms infer=${(tDone - tLoaded) / 1_000_000}ms " +
                "threads=${WhisperCpuConfig.preferredThreadCount}",
        )
        return cleanTranscript(raw)
    }

    /**
     * Preload the model context ahead of the first dictation (e.g. on service start) so
     * the first transcription doesn't pay the createContextFromFile cost. Best-effort:
     * returns quietly if the model isn't downloaded. Reuses the same cached ctx/loadedId
     * as [transcribe], so a warmed context is used directly by the next call.
     */
    suspend fun warm(context: Context, modelId: String) {
        val id = modelId.ifBlank { WhisperModelManager.DEFAULT_MODEL }
        if (!WhisperModelManager.isReady(context, id)) return
        loadLock.withLock {
            if (ctx != null && loadedId == id) return  // already warm
            val t0 = System.nanoTime()
            ctx?.let { runCatching { it.release() } }
            ctx = withContext(Dispatchers.IO) {
                WhisperContext.createContextFromFile(WhisperModelManager.modelFile(context, id).absolutePath)
            }
            loadedId = id
            Log.i("LocalWhisperStt", "warm model=$id load=${(System.nanoTime() - t0) / 1_000_000}ms")
        }
    }

    /**
     * Resolve the language requested in Settings. "keyboard" is best-effort: Android exposes
     * the current IME subtype, but some keyboards switch languages internally without updating
     * that subtype. In that case fall back to Whisper auto-detection.
     */
    private fun resolveLanguage(context: Context, setting: String): String? {
        if (setting in setOf("ru", "en", "de", "es")) return setting
        if (setting != "keyboard") return null

        val imm = context.getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
        val subtype = runCatching { imm?.currentInputMethodSubtype }.getOrNull() ?: return null
        val tag = if (android.os.Build.VERSION.SDK_INT >= 24) subtype.languageTag else ""
        val raw = tag.ifBlank { subtype.locale }
        val lang = raw.substringBefore('-').substringBefore('_').lowercase()
        return lang.takeIf { it in setOf("ru", "en", "de", "es") }
    }

    /** Strip whisper's bracketed non-speech markers (e.g. [BLANK_AUDIO], [Music]). */
    private fun cleanTranscript(s: String): String =
        s.replace(Regex("\\[[^\\]]*]"), "").trim()
}
