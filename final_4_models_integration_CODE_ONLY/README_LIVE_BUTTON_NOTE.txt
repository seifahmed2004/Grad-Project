This patch intentionally does not replace the full live page.
It only fixes the global persistent state bug that was restoring the key:
`live_stt_convert_button`.

If the live page still errors after applying this patch, remove explicit button keys in the live STT section:
- key="live_stt_convert_button"
- key="live_stt_clear_button"

The fixed persistent state no longer saves/restores those keys.
