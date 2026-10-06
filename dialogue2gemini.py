#!/usr/bin/env python3
"""
dialogue2gemini.py IN.json OUT.json

Универсальный инжектор: конвертирует нормализованный экспорт диалога
(*.dialogue.json из Алисы, GigaChat и др.) в формат лога Gemini AI Studio
(chunkedPrompt) и дописывает результат в OUT.json на месте (in-place, r+).

Вход (любой нормализованный плоский список реплик):
    [{"index": 0, "role": "user"|"assistant"|"model",
      "content_markdown": "...", "content_text": "...",
      "message_id": "<uuid>"}, ...]

Выход (Google AI Studio):
    {"runSettings": {...}, "systemInstruction": {...},
     "chunkedPrompt": {"chunks": [{"text","role","tokenCount","createTime",
                                    "parts","finishReason"}],
                        "pendingInputs": []}}

Поведение:
- Вход (IN.json) НИКОГДА не изменяется.
- Если OUT.json уже существует и это валидный Gemini-лог (например, «канарейка») —
  новые чанки дописываются в конец его chunkedPrompt.chunks, runSettings и
  systemInstruction берутся из самого OUT.json как есть.
- Если OUT.json не существует — создаётся с нуля с полным дефолтным
  runSettings (зашит в скрипт).
- OUT.json открывается строго в режиме "r+" (не удаляется и не пересоздаётся),
  содержимое переписывается на месте и обрезается truncate() — сохраняется тот же
  файл на диске, тот же inode и Google Drive file-id. Веб-клиент AI Studio
  подтягивает обновление на лету без срыва соединения.

Детали конвертации:
- role: assistant -> model, user -> user, model -> model
- text: берётся из content_markdown (fallback: content_text)
- createTime: если message_id — UUIDv7, метка времени извлекается из него
  (первые 48 бит = ms since epoch), иначе — текущее время + шаг 5 сек.
- tokenCount: оценка len(text)/3.3 (откалибровано по реальным логам Gemini
  AI Studio для кириллицы).
- role=model чанки получают "parts":[{"text":...}] и "finishReason":"STOP",
  как в нативных логах AI Studio.
"""
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

CHARS_PER_TOKEN = 3.3  # калибровка по реальным tokenCount для кириллицы
TIME_STEP_FALLBACK = timedelta(seconds=5)

DEFAULT_RUN_SETTINGS = {
    "temperature": 1.0,
    "model": "models/gemini-2.5-pro",
    "topP": 0.95,
    "topK": 64,
    "maxOutputTokens": 65536,
    "safetySettings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    ],
    "enableCodeExecution": False,
    "enableSearchAsATool": True,
    "enableBrowseAsATool": False,
    "googleSearch": {},
    "thinkingLevel": "THINKING_HIGH",
    "enableImageSearch": False,
    "enableGoogleMaps": False,
    "enableAgentThinkingSummariesControl": False,
    "enableAgentVisualizationControl": False,
    "enableAgentCollaborativePlanningControl": False,
    "environmentMode": "new",
}
DEFAULT_SYSTEM_INSTRUCTION = {"parts": [{"text": ""}]}


def decode_uuid7_time(message_id):
    try:
        u = uuid.UUID(message_id)
    except (ValueError, AttributeError, TypeError):
        return None
    if u.version != 7:
        return None
    return datetime.fromtimestamp((u.int >> 80) / 1000, tz=timezone.utc)


def iso_z(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def estimate_tokens(text):
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def map_role(role):
    return "model" if role in ("assistant", "model") else "user"


def build_chunks(dialogue_messages):
    chunks = []
    clock = datetime.now(timezone.utc)

    for msg in dialogue_messages:
        role = map_role(msg.get("role", "user"))
        text = msg.get("content_markdown") or msg.get("content_text") or ""

        dt = decode_uuid7_time(msg.get("message_id", ""))
        if dt is None:
            dt = clock
        clock = dt + TIME_STEP_FALLBACK

        chunk = {
            "text": text,
            "role": role,
            "tokenCount": estimate_tokens(text),
            "createTime": iso_z(dt),
        }
        if role == "model":
            chunk["finishReason"] = "STOP"
            chunk["parts"] = [{"text": text}]
        chunks.append(chunk)
    return chunks


def load_existing_output(path):
    """None если файла нет или он не похож на валидный Gemini-лог."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "chunkedPrompt" not in data:
        return None
    return data


def main():
    if len(sys.argv) != 3:
        sys.exit(f"Использование: {sys.argv[0]} IN.dialogue.json OUT.json")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, encoding="utf-8") as f:
        dialogue = json.load(f)
    if not isinstance(dialogue, list):
        sys.exit(f"Ожидался плоский список сообщений диалога, получено: {type(dialogue).__name__}")

    new_chunks = build_chunks(dialogue)

    existing = load_existing_output(out_path)
    if existing is not None:
        run_settings = existing["runSettings"]
        system_instruction = existing["systemInstruction"]
        all_chunks = existing["chunkedPrompt"]["chunks"] + new_chunks
        mode = "r+"
    else:
        run_settings = DEFAULT_RUN_SETTINGS
        system_instruction = DEFAULT_SYSTEM_INSTRUCTION
        all_chunks = new_chunks
        mode = "w"

    result = {
        "runSettings": run_settings,
        "systemInstruction": system_instruction,
        "chunkedPrompt": {"chunks": all_chunks, "pendingInputs": []},
    }

    # Открываем на дозапись и обрезаем на месте (не удаляем/пересоздаём файл) —
    # критически важно для сохранения дескриптора и связи с Google Drive / веб-клиентом.
    with open(out_path, mode, encoding="utf-8") as f:
        f.seek(0)
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.truncate()

    n_user = sum(1 for c in all_chunks if c["role"] == "user")
    n_model = sum(1 for c in all_chunks if c["role"] == "model")
    action = f"дописано {len(new_chunks)}" if existing is not None else f"создано {len(new_chunks)}"
    print(f"OK: {in_path} -> {out_path} ({action} сообщений)")
    print(f"  всего в chunkedPrompt: {len(all_chunks)} (user: {n_user}, model: {n_model})")


if __name__ == "__main__":
    main()