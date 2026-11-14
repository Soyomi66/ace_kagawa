from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ASR_MODEL = "conformer-unified-ja-JP-asr-streaming-asr-bls-ensemble"
EXPECTED_ASR_RMIR = "nvidia/riva/rmir_asr_conformer_unified_ja_jp_str:2.19.0"
EXPECTED_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
EXPECTED_LLM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
EXPECTED_RAG_SERVER_URL = "http://10.209.1.12:8081/v1"
EXPECTED_RAG_COLLECTION = "ace_kagawa"
EXPECTED_RAG_MAX_TOKENS = 64
EXPECTED_RAG_VDB_TOP_K = 12
EXPECTED_RAG_RERANKER_TOP_K = 5
EXPECTED_RAG_MULTIMODAL_RERANKER_TOP_K = 10
EXPECTED_RAG_MODE = "auto"
EXPECTED_RAG_ROUTE_KEYWORDS = ["論文", "資料", "EBC", "SiC/SiC"]
EXPECTED_KAGAWA_PROFILE_ROUTE_KEYWORDS = [
    "専門分野",
    "学歴",
    "職歴",
    "役職",
    "現職",
    "学位",
    "所属",
    "生年月日",
    "年齢",
    "誰ですか",
]
EXPECTED_EXPANDED_ROUTE_KEYWORDS = [
    "専攻",
    "学費",
    "入学金",
    "授業料",
    "奨学金",
    "学生支援",
    "研究",
    "スパコン",
    "スーパーコンピュータ",
    "青嵐",
    "SEIRAN",
    "DGX B200",
]
EXPECTED_RAG_PROVIDER = "local"
EXPECTED_LOCAL_RAG_CORPUS = "data/rag/corpus"
EXPECTED_LOCAL_RAG_DB = "data/rag/local/local_rag.sqlite"
EXPECTED_LOCAL_RAG_RUNTIME_DB = "/code/configs/local_rag.sqlite"
EXPECTED_LOCAL_RAG_TOP_K = 3
EXPECTED_LOCAL_RAG_MAX_CONTEXT_CHARS = 2800


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_tokkio_workspace", ROOT / "infra" / "tokkio" / "prepare_tokkio_workspace.py")
customize = load_module("customize_tokkio_japanese", ROOT / "infra" / "tokkio" / "customize_tokkio_japanese.py")
# The upstream bot.py lines that apply_patch rewrites.
UPSTREAM_BOT_FIXTURE = (
    "# from pipecat.audio.vad.silero import SileroVADAnalyzer"
    + customize.OLD_VAD_IMPORT_SNIPPET
    + customize.OLD_VAD_SNIPPET
    + customize.OLD_LLM_SNIPPET
    + "\n"
    + customize.OLD_BOT_SNIPPET
    + customize.OLD_CONTEXT_AGGREGATOR_SNIPPET
    + customize.OLD_PIPELINE_STT_SNIPPET
)


def load_generated_config_model() -> types.ModuleType:
    """Execute CONFIG_PY the way the controller imports src/config.py."""
    import pydantic

    animation_graph_module = types.ModuleType("nvidia_pipecat.services.animation_graph_service")

    class AnimationGraphConfiguration(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")

    animation_graph_module.AnimationGraphConfiguration = AnimationGraphConfiguration
    module = types.ModuleType("generated_tokkio_config")
    names = (
        "nvidia_pipecat",
        "nvidia_pipecat.services",
        "nvidia_pipecat.services.animation_graph_service",
        module.__name__,
    )
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules["nvidia_pipecat"] = types.ModuleType("nvidia_pipecat")
    sys.modules["nvidia_pipecat.services"] = types.ModuleType("nvidia_pipecat.services")
    sys.modules["nvidia_pipecat.services.animation_graph_service"] = animation_graph_module
    # pydantic resolves postponed annotations through sys.modules[cls.__module__].
    sys.modules[module.__name__] = module
    try:
        exec(compile(customize.CONFIG_PY, "config.py", "exec"), module.__dict__)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def install_stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    if "." not in name:
        module.__path__ = []
    sys.modules[name] = module
    return module


def load_tokkio_llm_with_stubs():
    class Logger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    class Frame:
        def __init__(self, content="", **kwargs):
            self.content = content
            self.__dict__.update(kwargs)

    class LLMFullResponseStartFrame(Frame):
        pass

    class LLMFullResponseEndFrame(Frame):
        pass

    class BaseLLMService:
        pass

    loguru = install_stub_module("loguru")
    loguru.logger = Logger()

    install_stub_module("nvidia_pipecat")
    install_stub_module("nvidia_pipecat.services")
    nvidia_llm = install_stub_module("nvidia_pipecat.services.nvidia_llm")
    nvidia_llm.NvidiaLLMService = BaseLLMService

    install_stub_module("pipecat")
    install_stub_module("pipecat.frames")
    frames = install_stub_module("pipecat.frames.frames")
    frames.LLMFullResponseEndFrame = LLMFullResponseEndFrame
    frames.LLMFullResponseStartFrame = LLMFullResponseStartFrame
    frames.TextFrame = Frame
    frames.TTSSpeakFrame = Frame
    install_stub_module("pipecat.processors")
    install_stub_module("pipecat.processors.aggregators")
    openai_context = install_stub_module("pipecat.processors.aggregators.openai_llm_context")
    openai_context.OpenAILLMContext = object
    install_stub_module("pipecat.services")
    install_stub_module("pipecat.services.openai")
    openai_llm = install_stub_module("pipecat.services.openai.llm")
    openai_llm.OpenAILLMService = BaseLLMService

    return load_module("tokkio_llm_fast_reply_test", ROOT / "infra" / "tokkio" / "tokkio_llm.py")


def load_tokkio_rag_with_stubs():
    """Load tokkio_rag.py as part of a stub package, as it is imported inside the controller."""

    tokkio_llm = load_tokkio_llm_with_stubs()

    class Frame:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.__dict__.update(kwargs)

    def identity_decorator(*args, **_kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda wrapped: wrapped

    install_stub_module("httpx")
    install_stub_module("nvidia_pipecat.frames")
    nvidia_rag_frames = install_stub_module("nvidia_pipecat.frames.nvidia_rag")
    nvidia_rag_frames.NvidiaRAGCitation = Frame
    nvidia_rag_frames.NvidiaRAGCitationsFrame = Frame
    nvidia_rag_service = install_stub_module("nvidia_pipecat.services.nvidia_rag")
    nvidia_rag_service.NvidiaRAGService = type("NvidiaRAGService", (), {})
    install_stub_module("nvidia_pipecat.utils")
    tracing = install_stub_module("nvidia_pipecat.utils.tracing")
    tracing.AttachmentStrategy = types.SimpleNamespace(NONE="none")
    tracing.traceable = identity_decorator
    tracing.traced = identity_decorator
    install_stub_module("openai")
    install_stub_module("openai.types")
    openai_chat = install_stub_module("openai.types.chat")
    openai_chat.ChatCompletionMessageParam = dict
    sys.modules["pipecat.frames.frames"].ErrorFrame = Frame

    package = install_stub_module("tokkio_rag_router_test_pkg")
    package.__path__ = []
    sys.modules["tokkio_rag_router_test_pkg.tokkio_llm"] = tokkio_llm
    load_module("tokkio_rag_router_test_pkg.local_rag", ROOT / "infra" / "rag" / "local_rag.py")
    load_module("tokkio_rag_router_test_pkg.tokkio_web_search", ROOT / "infra" / "tokkio" / "tokkio_web_search.py")
    return load_module("tokkio_rag_router_test_pkg.tokkio_rag", ROOT / "infra" / "tokkio" / "tokkio_rag.py")


class TokkioLlmConfigTests(unittest.TestCase):
    def test_resolve_llm_settings_defaults_to_nemotron_ultra_nim_endpoint(self) -> None:
        settings = prepare.resolve_llm_settings(
            {
                "TOKKIO_APP_HOST_IPV4_ADDR": "10.0.0.42",
                "TOKKIO_NVIDIA_API_KEY": "nvidia-key",
            }
        )

        self.assertEqual(settings.base_url, EXPECTED_LLM_BASE_URL)
        self.assertEqual(settings.model, EXPECTED_LLM_MODEL)
        self.assertEqual(settings.api_key, "nvidia-key")

    def test_nemotron_ultra_uses_realtime_non_reasoning_parameters(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        params = tokkio_llm.nvidia_model_input_params(EXPECTED_LLM_MODEL)

        self.assertEqual(params["temperature"], 0.0)
        self.assertEqual(params["max_tokens"], 512)
        self.assertEqual(
            params["extra"],
            {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            },
        )
        self.assertEqual(tokkio_llm.nvidia_model_input_params("mistralai/mistral-nemotron"), {})

    def test_gemma4_uses_the_evaluated_sampling_without_thinking(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        for model in (
            "gemma-4-12b-it",
            "google/gemma-4-12b-it",
            " Gemma-4-E4B-it ",
            "gemma-4-26b-a4b-it",
            "google/gemma-4-26B-A4B-it",
        ):
            with self.subTest(model=model):
                params = tokkio_llm.nvidia_model_input_params(model)

                self.assertEqual(params["temperature"], 0.1)
                self.assertEqual(params["top_p"], 0.95)
                self.assertEqual(params["max_tokens"], 512)
                self.assertEqual(
                    params["extra"],
                    {
                        "extra_body": {
                            "top_k": 64,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                    },
                )

    def test_swallow_keeps_the_server_default_sampling(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        self.assertEqual(tokkio_llm.nvidia_model_input_params("llama-3.1-swallow-8b-instruct-v0.5"), {})

    def test_rag_router_built_like_bot_py_sends_the_gemma4_sampling(self) -> None:
        # bot.py passes no params= and gives the router the RAG server's max_tokens (64);
        # the LLM request must still get the Gemma 4 sampling.
        tokkio_rag = load_tokkio_rag_with_stubs()
        base_service = sys.modules["pipecat.services.openai.llm"].OpenAILLMService

        def capture_init(self, **kwargs):
            self.init_kwargs = kwargs

        base_service.__init__ = capture_init
        base_service.InputParams = staticmethod(lambda **params: params)

        router = tokkio_rag.TokkioNvidiaLLMRAGRouterService(
            api_key="local-key",
            base_url="http://10.209.1.12:8011/v1",
            model="gemma-4-12b-it",
            collection_name="ace_kagawa",
            max_tokens=64,
            provider="nvidia",
            route_keywords=["研究"],
            filler=["確認しています"],
            time_delay=8.0,
        )

        params = router.init_kwargs["params"]
        self.assertEqual(params["temperature"], 0.1)
        self.assertEqual(params["max_tokens"], 512)
        self.assertEqual(params["extra"]["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(router.max_tokens, 64)

    def test_generated_env_exports_llm_api_key_for_local_endpoint(self) -> None:
        generated = prepare.build_generated_env(
            {
                "TOKKIO_NVIDIA_API_KEY": "real-nvidia-key",
                "TOKKIO_NGC_CLI_API_KEY": "real-ngc-key",
                "TOKKIO_OPENAI_API_KEY": "",
                "TOKKIO_APP_HOST_IPV4_ADDR": "10.0.0.42",
                "TOKKIO_APP_HOST_SSH_USER": "kyano",
                "TOKKIO_LLM_API_KEY": "local-key",
            }
        )

        self.assertIn('export NVIDIA_LLM_API_KEY="local-key"', generated)

    def test_generated_env_reuses_nvidia_api_key_for_hosted_nim(self) -> None:
        generated = prepare.build_generated_env(
            {
                "TOKKIO_NVIDIA_API_KEY": "real-nvidia-key",
                "TOKKIO_NGC_CLI_API_KEY": "real-ngc-key",
                "TOKKIO_OPENAI_API_KEY": "",
                "TOKKIO_APP_HOST_IPV4_ADDR": "10.0.0.42",
                "TOKKIO_APP_HOST_SSH_USER": "kyano",
            }
        )

        self.assertIn('export NVIDIA_LLM_API_KEY="real-nvidia-key"', generated)

    def test_resolve_rag_settings_defaults_to_disabled_host_side_service(self) -> None:
        settings = prepare.resolve_rag_settings({"TOKKIO_APP_HOST_IPV4_ADDR": "10.209.1.12"})

        self.assertFalse(settings.enabled)
        self.assertEqual(settings.server_url, EXPECTED_RAG_SERVER_URL)
        self.assertEqual(settings.collection_name, EXPECTED_RAG_COLLECTION)
        self.assertTrue(settings.use_knowledge_base)
        self.assertEqual(settings.max_tokens, EXPECTED_RAG_MAX_TOKENS)
        self.assertEqual(settings.vdb_top_k, EXPECTED_RAG_VDB_TOP_K)
        self.assertEqual(settings.reranker_top_k, EXPECTED_RAG_RERANKER_TOP_K)
        self.assertEqual(settings.multimodal_reranker_top_k, EXPECTED_RAG_MULTIMODAL_RERANKER_TOP_K)
        self.assertTrue(settings.enable_reranker)
        self.assertEqual(settings.mode, EXPECTED_RAG_MODE)
        self.assertIn("論文", settings.route_keywords)
        self.assertIn("SiC/SiC", settings.route_keywords)
        for keyword in ("研究", "プロジェクト", "専攻", "学費", "奨学金", "学生支援"):
            self.assertIn(keyword, settings.route_keywords)
        for keyword in EXPECTED_KAGAWA_PROFILE_ROUTE_KEYWORDS:
            self.assertIn(keyword, settings.route_keywords)
        self.assertFalse(settings.fallback_to_llm_on_error)
        self.assertEqual(settings.provider, EXPECTED_RAG_PROVIDER)
        self.assertEqual(settings.local_corpus_path, EXPECTED_LOCAL_RAG_CORPUS)
        self.assertEqual(settings.local_db_path, EXPECTED_LOCAL_RAG_DB)
        self.assertEqual(settings.local_runtime_db_path, EXPECTED_LOCAL_RAG_RUNTIME_DB)
        self.assertEqual(settings.local_top_k, EXPECTED_LOCAL_RAG_TOP_K)
        self.assertEqual(settings.local_max_context_chars, EXPECTED_LOCAL_RAG_MAX_CONTEXT_CHARS)

    def test_resolve_rag_settings_normalizes_url_and_bool_values(self) -> None:
        settings = prepare.resolve_rag_settings(
            {
                "TOKKIO_RAG_ENABLED": "true",
                "TOKKIO_RAG_PROVIDER": "nvidia",
                "TOKKIO_RAG_SERVER_URL": "http://192.0.2.10:8081",
                "TOKKIO_RAG_COLLECTION_NAME": "manuals",
                "TOKKIO_RAG_USE_KNOWLEDGE_BASE": "false",
                "TOKKIO_RAG_MAX_TOKENS": "512",
                "TOKKIO_RAG_VDB_TOP_K": "16",
                "TOKKIO_RAG_RERANKER_TOP_K": "6",
                "TOKKIO_RAG_MULTIMODAL_RERANKER_TOP_K": "10",
                "TOKKIO_RAG_ENABLE_RERANKER": "false",
                "TOKKIO_RAG_MODE": "always",
                "TOKKIO_RAG_ROUTE_KEYWORDS": "論文, 資料, EBC, SiC/SiC",
                "TOKKIO_RAG_FALLBACK_TO_LLM_ON_ERROR": "false",
                "TOKKIO_LOCAL_RAG_DB": "/tmp/custom-local.sqlite",
                "TOKKIO_LOCAL_RAG_RUNTIME_DB_PATH": "/code/configs/custom-local.sqlite",
                "TOKKIO_LOCAL_RAG_TOP_K": "4",
                "TOKKIO_LOCAL_RAG_MAX_CONTEXT_CHARS": "1200",
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.mode, "always")
        self.assertEqual(settings.provider, "nvidia")
        self.assertEqual(settings.server_url, "http://192.0.2.10:8081/v1")
        self.assertEqual(settings.collection_name, "manuals")
        self.assertFalse(settings.use_knowledge_base)
        self.assertEqual(settings.max_tokens, 512)
        self.assertEqual(settings.vdb_top_k, 16)
        self.assertEqual(settings.reranker_top_k, 6)
        self.assertEqual(settings.multimodal_reranker_top_k, 10)
        self.assertFalse(settings.enable_reranker)
        self.assertEqual(settings.route_keywords, EXPECTED_RAG_ROUTE_KEYWORDS)
        self.assertFalse(settings.fallback_to_llm_on_error)
        self.assertEqual(settings.local_db_path, "/tmp/custom-local.sqlite")
        self.assertEqual(settings.local_runtime_db_path, "/code/configs/custom-local.sqlite")
        self.assertEqual(settings.local_top_k, 4)
        self.assertEqual(settings.local_max_context_chars, 1200)

    def test_resolve_rag_settings_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "TOKKIO_RAG_MODE"):
            prepare.resolve_rag_settings({"TOKKIO_RAG_ENABLED": "true", "TOKKIO_RAG_MODE": "sometimes"})

    def test_resolve_rag_settings_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "TOKKIO_RAG_PROVIDER"):
            prepare.resolve_rag_settings({"TOKKIO_RAG_ENABLED": "true", "TOKKIO_RAG_PROVIDER": "remote"})

    def test_prepare_freshness_gate_rejects_changed_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            source_path = corpus_dir / "school.md"
            source_path.write_text("# School\n\n東京工科大学の概要です。\n", encoding="utf-8")
            db_path = Path(tmp) / "local_rag.sqlite"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "infra" / "rag" / "build_local_index.py"),
                    "--corpus",
                    str(corpus_dir),
                    "--db",
                    str(db_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            settings = prepare.resolve_rag_settings(
                {
                    "TOKKIO_RAG_ENABLED": "true",
                    "TOKKIO_RAG_PROVIDER": "local",
                    "TOKKIO_LOCAL_RAG_CORPUS": str(corpus_dir),
                    "TOKKIO_LOCAL_RAG_DB": str(db_path),
                }
            )

            self.assertTrue(prepare.verify_local_rag_index(settings)["fresh"])
            source_path.write_text("# School\n\n東京工科大学の概要を更新しました。\n", encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError):
                prepare.verify_local_rag_index(settings)

    def test_custom_config_yaml_uses_nemotron_ultra_nim_endpoint_and_model(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn(f'base_url: "{EXPECTED_LLM_BASE_URL}"', yaml_text)
        self.assertIn(f'model: "{EXPECTED_LLM_MODEL}"', yaml_text)
        self.assertIn("標準語で自然かつ簡潔", yaml_text)
        self.assertIn("40から60文字", yaml_text)
        self.assertIn("原則1文", yaml_text)
        self.assertIn("100文字以内", yaml_text)
        self.assertIn("最大2文", yaml_text)
        self.assertNotIn("80文字以内", yaml_text)
        self.assertIn("time_delay: 1.0", yaml_text)
        self.assertIn('- "そうですね。"', yaml_text)
        self.assertNotIn("少々お待ちください", yaml_text)
        self.assertIn('llm_processor: "NvidiaLLMService"', yaml_text)

    def test_custom_config_yaml_can_enable_rag_processor(self) -> None:
        yaml_text = customize.build_config_yaml(
            rag_enabled=True,
            rag_mode="always",
            rag_server_url="http://10.209.1.12:8081",
            rag_collection_name="ace_kagawa",
            rag_max_tokens=512,
        )

        self.assertIn('llm_processor: "NvidiaRAGService"', yaml_text)
        self.assertIn(f'rag_server_url: "{EXPECTED_RAG_SERVER_URL}"', yaml_text)
        self.assertIn(f'collection_name: "{EXPECTED_RAG_COLLECTION}"', yaml_text)
        self.assertIn("max_tokens: 512", yaml_text)
        self.assertIn(f"vdb_top_k: {EXPECTED_RAG_VDB_TOP_K}", yaml_text)
        self.assertIn(f"reranker_top_k: {EXPECTED_RAG_RERANKER_TOP_K}", yaml_text)
        self.assertIn(f"multimodal_reranker_top_k: {EXPECTED_RAG_MULTIMODAL_RERANKER_TOP_K}", yaml_text)
        self.assertIn("enable_reranker: true", yaml_text)

    def test_custom_config_yaml_can_enable_auto_rag_router(self) -> None:
        yaml_text = customize.build_config_yaml(
            rag_enabled=True,
            rag_mode="auto",
            rag_route_keywords=EXPECTED_RAG_ROUTE_KEYWORDS,
        )

        self.assertIn('llm_processor: "NvidiaLLMRAGRouterService"', yaml_text)
        self.assertIn("NvidiaRAGRouterService:", yaml_text)
        self.assertIn('provider: "local"', yaml_text)
        self.assertIn(f'local_db_path: "{EXPECTED_LOCAL_RAG_RUNTIME_DB}"', yaml_text)
        self.assertIn(f"local_top_k: {EXPECTED_LOCAL_RAG_TOP_K}", yaml_text)
        self.assertIn(f"local_max_context_chars: {EXPECTED_LOCAL_RAG_MAX_CONTEXT_CHARS}", yaml_text)
        route_section = yaml_text.split("    route_keywords:\n", 1)[1].split(
            "    fallback_to_llm_on_error:", 1
        )[0]
        for keyword in EXPECTED_RAG_ROUTE_KEYWORDS:
            self.assertIn(f'        - "{keyword}"', route_section)
        self.assertNotIn('        - "香川先生"', route_section)
        self.assertNotIn(customize.RAG_ROUTE_KEYWORDS_TEMPLATE_MARKER, yaml_text)
        self.assertIn("fallback_to_llm_on_error: false", yaml_text)

    def test_grounded_rag_failures_never_use_direct_llm_fallback(self) -> None:
        source = (ROOT / "infra" / "tokkio" / "tokkio_rag.py").read_text(encoding="utf-8")
        local_rag_block = source.split(
            "    async def _stream_local_rag_response", 1
        )[1].split("    async def _stream_rag_response", 1)[0]
        router_error_block = source.split(
            '            logger.error(f"RAG router failed, Error: {exc!r}")', 1
        )[1]

        self.assertIn("GROUNDED_ANSWER_UNAVAILABLE_REPLY", source)
        # An empty prompt context still gets the fixed reply.
        self.assertEqual(local_rag_block.count("await self._push_grounded_answer_unavailable()"), 1)
        # Rejected evidence goes to the LLM, always with the note that says what not to make up.
        self.assertIn(
            "self._build_local_rag_context(context, ungrounded_prompt_note(assessment), query)",
            local_rag_block,
        )
        # The visitor's turn never reaches the LLM bare, and a failed grounded path never reaches it.
        self.assertNotIn("await super()._process_context(context)", local_rag_block)
        self.assertNotIn("await super()._process_context(context)", router_error_block)
        self.assertIn(
            "await self._push_grounded_answer_unavailable(start_metrics=False)",
            source,
        )
        self.assertIn(
            "await self.push_frame(TTSSpeakFrame(GROUNDED_ANSWER_UNAVAILABLE_REPLY))",
            router_error_block,
        )

    def test_rag_router_lets_the_llm_answer_rejected_questions_with_a_note(self) -> None:
        import asyncio
        from unittest import mock

        tokkio_rag = load_tokkio_rag_with_stubs()
        local_rag = sys.modules["tokkio_rag_router_test_pkg.local_rag"]

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "kagawa.md").write_text(
                "# Kagawa\n\n"
                "## kaken_keywords\n\n"
                "source_url: https://nrid.nii.ac.jp/example\n"
                "source_priority: P1\n"
                "record_type: kagawa_research_keywords\n"
                "chunk_id: chunk_kagawa_kaken_keywords\n"
                "keywords: 複合材料, CMC, SiC/SiC\n\n"
                "香川豊先生の研究分野は、複合材料、セラミックス基複合材料（CMC）、界面力学特性です。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"
            local_rag.build_index(corpus_dir, db_path)

            router = object.__new__(tokkio_rag.TokkioNvidiaLLMRAGRouterService)
            router.provider = "local"
            router.local_db_path = str(db_path)
            router.local_top_k = 3
            router.local_max_context_chars = 2800
            router.enable_citations = False
            router.fallback_to_llm_on_error = False
            router.route_keywords = tuple(customize.DEFAULT_RAG_ROUTE_KEYWORDS)
            pushed: list = []
            sent_to_llm: list = []

            async def noop(*_args, **_kwargs) -> None:
                return None

            async def push_frame(frame, *_args) -> None:
                pushed.append(frame)

            async def direct_llm(_self, context) -> None:
                sent_to_llm.append(context.get_messages()[-1]["content"])

            router.start_ttfb_metrics = noop
            router.stop_ttfb_metrics = noop
            router.push_error = noop
            router.push_frame = push_frame

            def run(text: str) -> None:
                pushed.clear()
                sent_to_llm.clear()
                turn = [{"role": "user", "content": text}]
                context = types.SimpleNamespace(get_messages=lambda: turn)
                with mock.patch.object(tokkio_rag, "_push_local_reply_if_available", return_value=False), \
                        mock.patch.object(tokkio_rag.TokkioNvidiaLLMService, "_process_context", direct_llm, create=True):
                    asyncio.run(router._process_context(context))

            # The corpus says nothing about his hobbies: no fixed sentence, the LLM answers
            # with the note for questions about Kagawa himself.
            run("香川先生の趣味は何ですか")
            self.assertEqual(pushed, [])
            self.assertEqual(len(sent_to_llm), 1)
            self.assertTrue(sent_to_llm[0].startswith(local_rag.UNGROUNDED_SELF_PROMPT_NOTE))
            self.assertTrue(sent_to_llm[0].endswith("香川先生の趣味は何ですか"))

            run("香川先生と山田先生の研究について教えてください")
            self.assertEqual(pushed, [])
            self.assertIn("「山田」という方", sent_to_llm[0])

            # A broken index is not answered from the LLM's own knowledge.
            router.local_db_path = str(Path(tmp) / "missing.sqlite")
            run("香川先生の趣味は何ですか")
            self.assertEqual(sent_to_llm, [])
            self.assertEqual(
                [frame.content for frame in pushed if getattr(frame, "content", "")],
                [tokkio_rag.GROUNDED_ANSWER_UNAVAILABLE_REPLY],
            )

    def test_rag_router_sends_generic_topics_outside_the_corpus_to_direct_llm(self) -> None:
        tokkio_rag = load_tokkio_rag_with_stubs()
        local_rag = sys.modules["tokkio_rag_router_test_pkg.local_rag"]

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "kagawa.md").write_text(
                "# Kagawa\n\n"
                "## kaken_keywords\n\n"
                "source_url: https://nrid.nii.ac.jp/example\n"
                "source_priority: P1\n"
                "record_type: kagawa_research_keywords\n"
                "chunk_id: chunk_kagawa_kaken_keywords\n"
                "keywords: 複合材料, CMC, SiC/SiC\n\n"
                "香川豊先生の研究分野は、複合材料、セラミックス基複合材料（CMC）、界面力学特性です。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"
            local_rag.build_index(corpus_dir, db_path)

            router = object.__new__(tokkio_rag.TokkioNvidiaLLMRAGRouterService)
            router.provider = "local"
            router.local_db_path = str(db_path)
            router.local_top_k = 3
            router.route_keywords = tuple(customize.DEFAULT_RAG_ROUTE_KEYWORDS)

            def routes_to_rag(text: str) -> bool:
                return router._should_use_rag([{"role": "user", "content": text}])

            # Routed only by the generic word 研究, and the corpus never mentions the topic.
            self.assertFalse(routes_to_rag("量子コンピュータの研究について教えてください"))
            quantum_turn = [{"role": "user", "content": "量子コンピュータの研究について教えてください"}]
            self.assertEqual(router._rag_route(quantum_turn), "outside_corpus")
            # The plain LLM receives the no-evidence note ahead of the user's words.
            context = types.SimpleNamespace(get_messages=lambda: quantum_turn)
            note_context = router._build_local_rag_context(
                context, tokkio_rag.OUTSIDE_CORPUS_PROMPT_NOTE, quantum_turn[0]["content"]
            )
            sent = note_context.get_messages()[-1]["content"]
            self.assertTrue(sent.startswith(tokkio_rag.OUTSIDE_CORPUS_PROMPT_NOTE))
            self.assertTrue(sent.endswith("量子コンピュータの研究について教えてください"))
            self.assertEqual(router._rag_route([{"role": "user", "content": "今日の天気は"}]), "direct")
            # Generic wording whose topic the corpus does cover stays grounded.
            self.assertTrue(routes_to_rag("セラミックスの研究について教えてください"))
            # Questions addressed to Kagawa stay on the grounded path even when unanswerable,
            # so the LLM answers them with the note that forbids inventing his private facts.
            self.assertTrue(routes_to_rag("香川先生の趣味は何ですか"))
            self.assertTrue(routes_to_rag("あなたの研究について教えてください"))

            # A broken index must reach the grounded path's error handler, not escape routing.
            router.local_db_path = str(Path(tmp) / "missing.sqlite")
            self.assertTrue(routes_to_rag("量子コンピュータの研究について教えてください"))

    def test_rag_router_asks_again_on_garbled_turns_and_notes_university_life_topics(self) -> None:
        import asyncio
        from unittest import mock

        tokkio_rag = load_tokkio_rag_with_stubs()
        router = object.__new__(tokkio_rag.TokkioNvidiaLLMRAGRouterService)
        router.provider = "nvidia"  # skip the corpus check; only the classifier routes
        router.route_keywords = tuple(customize.DEFAULT_RAG_ROUTE_KEYWORDS)
        pushed: list = []
        sent_to_llm: list = []

        async def noop() -> None:
            return None

        async def push_frame(frame, *_args) -> None:
            pushed.append(frame)

        async def direct_llm(_self, context) -> None:
            sent_to_llm.append(context.get_messages()[-1]["content"])

        router.start_ttfb_metrics = noop
        router.stop_ttfb_metrics = noop
        router.push_frame = push_frame

        def run(text: str) -> None:
            pushed.clear()
            sent_to_llm.clear()
            turn = [{"role": "user", "content": text}]
            context = types.SimpleNamespace(get_messages=lambda: turn)
            with mock.patch.object(tokkio_rag, "_push_local_reply_if_available", return_value=False), \
                    mock.patch.object(tokkio_rag.TokkioNvidiaLLMService, "_process_context", direct_llm, create=True):
                asyncio.run(router._process_context(context))

        run("千五百一二百話")
        texts = [frame.content for frame in pushed if getattr(frame, "content", "")]
        self.assertEqual(texts, [tokkio_rag.ASK_AGAIN_REPLY])
        self.assertEqual(sent_to_llm, [])

        run("学食はおいしいですか")
        self.assertEqual(len(sent_to_llm), 1)
        self.assertTrue(sent_to_llm[0].startswith(tokkio_rag.OUTSIDE_CORPUS_PROMPT_NOTE))

        run("今日の天気は")
        self.assertEqual(sent_to_llm, ["今日の天気は"])

        # A question that mentions the campus gets the note as well: no made-up shops or hours.
        run("大学の近くにコンビニはありますか")
        self.assertTrue(sent_to_llm[0].startswith(tokkio_rag.OUTSIDE_CORPUS_PROMPT_NOTE))
        run("おすすめの本を教えてください")
        self.assertEqual(sent_to_llm, ["おすすめの本を教えてください"])

    def test_web_search_query_sends_general_questions_only(self) -> None:
        load_tokkio_rag_with_stubs()
        web = sys.modules["tokkio_rag_router_test_pkg.tokkio_web_search"]

        sent = {
            "えっと、今日の天気を教えてください": "八王子 今日の天気",
            "蒲田の天気はどうですか": "蒲田の天気はどう",
            "セラミックス基複合材料とは何ですか": "セラミックス基複合材料とは",
            "AIとは何ですか": "AIとは",
            "量子コンピュータについて教えてください": "量子コンピュータとは",
            "2025年のノーベル化学賞は誰ですか": "2025年のノーベル化学賞は誰",
            "富士山の高さはどのくらいですか": "富士山の高さはどのくらい",
        }
        for text, query in sent.items():
            with self.subTest(text=text):
                self.assertEqual(web.web_search_query(text), query)
        # The router marks topics outside the corpus; those need no question pattern.
        self.assertIsNone(web.web_search_query("学食はおいしいですか"))
        self.assertEqual(web.web_search_query("学食はおいしいですか", topic_outside_corpus=True), "学食はおいしい")
        self.assertEqual(web.web_search_query("学食はありますか", topic_outside_corpus=True), "学食")
        self.assertEqual(
            web.web_search_query("学食はありますか", topic_outside_corpus=True, about_university=True),
            "東京工科大学 学食",
        )
        # The excerpts are not this university's documents: other universities' numbers are not ours.
        self.assertIn("他の大学や他の施設の数値や事実を、東京工科大学や私のこととして答えず", web.WEB_SEARCH_PROMPT_NOTE)
        for text in (
            # The persona, named people, personal details.
            "あなたの趣味は何ですか",
            "香川先生の経歴を教えてください",
            "山田さんはどこにいますか",
            "佐藤教授の研究室はどこですか",
            "私の電話番号は09012345678です",
            "090-1234-5678に天気を教えてください",
            "０９０ １２３４ ５６７８の天気はどうですか",
            "娘の学校の天気はどうですか",
            # Questions to the persona without a subject.
            "趣味は何ですか",
            "どこの出身ですか",
            "好きな食べ物は何ですか",
            "名前は?",
            # Back-channels, requests, follow-ups that point back.
            "そうなんですか",
            "聞こえますか",
            # ことは is not とは: this asks about the persona's experience.
            "ChatGPTを使ったことはありますか",
            "ちょっと待ってください",
            "もう少し詳しく教えてください",
            "それはいつですか",
            "その研究はいつ始まりましたか",
            "こんにちは",
            "ありがとうございました",
            "はい、お願いします",
            "なるほど",
            # English mode stays local.
            "What is AI?",
        ):
            with self.subTest(text=text):
                self.assertIsNone(web.web_search_query(text))
        # The persona and personal-detail checks hold even for topics outside the corpus.
        for text in ("あなたは寮に住んでいましたか", "山田さんは学食に行きますか", "私の住所は寮です"):
            with self.subTest(text=text):
                self.assertIsNone(web.web_search_query(text, topic_outside_corpus=True))

    def test_search_breaker_pauses_after_consecutive_failures(self) -> None:
        load_tokkio_rag_with_stubs()
        web = sys.modules["tokkio_rag_router_test_pkg.tokkio_web_search"]
        breaker = web.SearchBreaker(max_failures=2, pause_s=300.0)

        breaker.record(False, 0.0)
        self.assertFalse(breaker.is_open(1.0))
        breaker.record(True, 1.0)
        breaker.record(False, 2.0)
        self.assertFalse(breaker.is_open(3.0))
        breaker.record(False, 3.0)
        self.assertTrue(breaker.is_open(4.0))
        self.assertFalse(breaker.is_open(303.1))

    def test_exa_search_sends_lite_japan_request_and_keeps_highlights(self) -> None:
        import asyncio

        load_tokkio_rag_with_stubs()
        web = sys.modules["tokkio_rag_router_test_pkg.tokkio_web_search"]
        calls = []

        async def fake_post(url, body, headers, timeout_s):
            calls.append((url, body, headers, timeout_s))
            return {
                "results": [
                    {"title": "就職状況", "url": "https://example.invalid/a", "publishedDate": "2026-07-24T00:00:00.000Z",
                     "highlights": ["就職率 98.1%", "  "]},
                    {"title": "空", "url": "https://example.invalid/b", "highlights": []},
                ]
            }

        searcher = web.ExaWebSearch("test-key", timeout_s=2.0, post_json=fake_post)
        results = asyncio.run(searcher.search("東京工科大学 就職率"))

        url, body, headers, timeout_s = calls[0]
        self.assertEqual(url, "https://api.exa.ai/search")
        self.assertEqual(body["type"], "lite")
        self.assertEqual(body["userLocation"], "JP")
        self.assertEqual(body["query"], "東京工科大学 就職率")
        self.assertEqual(headers["x-api-key"], "test-key")
        self.assertEqual(timeout_s, 2.0)
        self.assertEqual([(r.title, r.published_date, r.text) for r in results], [("就職状況", "2026-07-24", "就職率 98.1%")])
        prompt = web.format_web_results_for_prompt(results, today=__import__("datetime").date(2026, 11, 13))
        self.assertTrue(prompt.startswith("参考情報: "))
        self.assertIn("2026-11-13", prompt)
        self.assertIn("[1] 就職状況: 就職率 98.1%", prompt)
        self.assertNotIn("2026-07-24", prompt)
        self.assertNotIn("example.invalid", prompt)

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(web.ExaWebSearch.from_key_file(Path(tmp) / "missing"))
            (Path(tmp) / "empty").write_text("\n", encoding="utf-8")
            self.assertIsNone(web.ExaWebSearch.from_key_file(Path(tmp) / "empty"))

    def test_rag_router_adds_web_results_for_general_questions_only(self) -> None:
        import asyncio
        from unittest import mock

        tokkio_rag = load_tokkio_rag_with_stubs()
        web = sys.modules["tokkio_rag_router_test_pkg.tokkio_web_search"]
        router = object.__new__(tokkio_rag.TokkioNvidiaLLMRAGRouterService)
        router.provider = "nvidia"
        router.route_keywords = tuple(customize.DEFAULT_RAG_ROUTE_KEYWORDS)
        router.filler = []
        router.time_delay = 1.0
        queries = []

        class FakeSearcher:
            timeout_s = 2.0

            async def search(self, query):
                queries.append(query)
                return [web.WebSearchResult("天気", "https://example.invalid", "", "八王子は晴れ、最高19度")]

        router._cached_web_searcher = FakeSearcher()
        web.SEARCH_BREAKER = web.SearchBreaker()
        tokkio_rag.SEARCH_BREAKER = web.SEARCH_BREAKER
        sent_to_llm = []

        async def noop() -> None:
            return None

        async def push_frame(frame, *_args) -> None:
            return None

        async def direct_llm(_self, context) -> None:
            sent_to_llm.append(context.get_messages()[-1]["content"])

        router.start_ttfb_metrics = noop
        router.stop_ttfb_metrics = noop
        router.push_frame = push_frame

        def run(text: str) -> str:
            sent_to_llm.clear()
            turn = [{"role": "user", "content": text}]
            context = types.SimpleNamespace(get_messages=lambda: turn)
            with mock.patch.object(tokkio_rag, "_push_local_reply_if_available", return_value=False), \
                    mock.patch.object(tokkio_rag.TokkioNvidiaLLMService, "_process_context", direct_llm, create=True):
                asyncio.run(router._process_context(context))
            return sent_to_llm[0]

        sent = run("今日の天気は")
        self.assertEqual(queries, ["八王子 今日の天気"])
        self.assertTrue(sent.startswith("参考情報: 以下はWeb検索"))
        self.assertIn("八王子は晴れ、最高19度", sent)

        queries.clear()
        self.assertEqual(run("あなたの趣味は何ですか"), "あなたの趣味は何ですか")
        self.assertEqual(queries, [])
        # A campus question gets the outside-corpus note, but that alone sends nothing to the web.
        self.assertTrue(run("大学の近くにコンビニはありますか").startswith(tokkio_rag.OUTSIDE_CORPUS_PROMPT_NOTE))
        self.assertEqual(queries, [])

        # University life is outside the corpus, so it is searched without a question pattern.
        # It is about this university, so its name goes into the search words.
        self.assertIn("八王子は晴れ", run("学食はおいしいですか"))
        self.assertEqual(queries, ["東京工科大学 学食はおいしい"])

        # A failed search falls back to the plain path (with the outside-corpus note when it applies).
        class FailingSearcher(FakeSearcher):
            async def search(self, query):
                raise TimeoutError("slow")

        router._cached_web_searcher = FailingSearcher()
        self.assertTrue(run("学食はおいしいですか").startswith(tokkio_rag.OUTSIDE_CORPUS_PROMPT_NOTE))
        # The second failure pauses searching; the next turn does not wait for the web.
        self.assertTrue(run("学食はおいしいですか").startswith(tokkio_rag.OUTSIDE_CORPUS_PROMPT_NOTE))
        self.assertTrue(tokkio_rag.SEARCH_BREAKER.is_open(__import__("time").time()))

        # No key file: web search stays off.
        del router._cached_web_searcher
        with tempfile.TemporaryDirectory() as tmp:
            router.local_db_path = str(Path(tmp) / "local_rag.sqlite")
            self.assertIsNone(router._web_searcher())

    def test_prepare_writes_the_web_search_key_file_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated_dir = Path(tmp)
            key_path = generated_dir / "exa_api_key"
            values = {"TOKKIO_ACE_BRANCH": "5.0.0-ga", "TOKKIO_EXA_API_KEY": "test-key"}

            result = prepare.sync_web_search_key(values, generated_dir)
            self.assertEqual(result, {"enabled": True, "provider": "exa", "reason": ""})
            self.assertEqual(key_path.read_text(encoding="utf-8"), "test-key\n")
            self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("test-key", repr(result))

            result = prepare.sync_web_search_key({**values, "TOKKIO_WEB_SEARCH": "off"}, generated_dir)
            self.assertFalse(result["enabled"])
            self.assertFalse(key_path.exists())

            prepare.sync_web_search_key(values, generated_dir)
            prepare.sync_web_search_key({"TOKKIO_ACE_BRANCH": "5.0.0-ga"}, generated_dir)
            self.assertFalse(key_path.exists())

    def test_custom_config_yaml_uses_early_cached_filler_delay(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn('    filler:\n        - "そうですね。"\n    time_delay: 1.0\n', yaml_text)
        self.assertIn("    filler: []", customize.build_config_yaml(tts_processor="QwenTTSService"))

    def test_custom_config_yaml_routes_kagawa_profile_detail_questions_by_default(self) -> None:
        yaml_text = customize.build_config_yaml(rag_enabled=True, rag_mode="auto")

        for keyword in EXPECTED_KAGAWA_PROFILE_ROUTE_KEYWORDS:
            self.assertIn(f'        - "{keyword}"', yaml_text)
        for keyword in EXPECTED_EXPANDED_ROUTE_KEYWORDS:
            self.assertIn(f'        - "{keyword}"', yaml_text)

    def test_custom_config_yaml_can_force_rag_off_even_when_enabled(self) -> None:
        yaml_text = customize.build_config_yaml(rag_enabled=True, rag_mode="off")

        self.assertIn('llm_processor: "NvidiaLLMService"', yaml_text)

    def test_tokkio_llm_fast_replies_cover_greetings_name_and_current_age(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        self.assertEqual(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "おはようございます。"}]),
            "おはようございます。お会いできてうれしいです。",
        )
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "先生のお名前を教えてください"}]),
            "私は香川豊です。",
        )
        # Someone else's name or age is not the persona's (2026-10-17 audit).
        for text in (
            "あなたのお父さんは何歳ですか。",
            "あなたの会社の名前は",
            "香川先生の名前の由来は",
            "息子は何歳ですか",
        ):
            with self.subTest(text=text):
                self.assertIsNone(tokkio_llm.get_fast_profile_reply([{"role": "user", "content": text}]))
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "あなたの名前は？"}]),
            "私は香川豊です。",
        )
        self.assertIsNone(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "香川先生の論文の名前は？"}])
        )
        self.assertIsNone(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "香川先生は誰ですか"}])
        )
        self.assertIsNone(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "香川先生の名前と専門分野は？"}])
        )
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply(
                [{"role": "user", "content": "香川先生の年齢は？"}],
                today=__import__("datetime").date(2026, 7, 19),
            ),
            "私は1952年9月19日生まれで、現在73歳です。",
        )
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply(
                [{"role": "user", "content": "香川先生は何歳ですか？"}],
                today=__import__("datetime").date(2026, 9, 19),
            ),
            "私は1952年9月19日生まれで、現在74歳です。",
        )
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply(
                [{"role": "user", "content": "先生はおいくつですか？"}],
                today=__import__("datetime").date(2026, 7, 19),
            ),
            "私は1952年9月19日生まれで、現在73歳です。",
        )

    def test_tokkio_llm_open_campus_greeting_uses_explicit_cues_and_full_script(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        script = tokkio_llm.get_scripted_speech_reply(
            [{"role": "user", "content": "オープンキャンパスのご挨拶をお願いします。"}]
        )

        self.assertEqual(script, tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT)
        self.assertEqual(script[0], "皆さん、こんにちは。")
        self.assertIn("USJとほぼ同じ広さ", "".join(script))
        self.assertIn("私はAI学長です。", "".join(script))
        self.assertEqual(
            script[-1],
            "ここで、開発した学生と私とのリアルなやり取りをご覧ください。",
        )
        self.assertIsNone(
            tokkio_llm.get_scripted_speech_reply(
                [{"role": "user", "content": "オープンキャンパスはいつですか？"}]
            )
        )

    def test_tokkio_llm_symposium_opening_uses_tts_sentence_segments(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        script = tokkio_llm.get_scripted_speech_reply(
            [{"role": "user", "content": "シンポジウムのオープニングをお願いします。"}]
        )

        self.assertEqual(script, tokkio_llm.SYMPOSIUM_OPENING_SCRIPT)
        self.assertEqual(len(script), 7)
        self.assertEqual(script[0], "Good afternoon, everyone.")
        self.assertEqual(
            script[-1],
            "And now, please welcome Professor Kagawa, the President of the Tokyo University of Technology.",
        )
        self.assertEqual(
            tokkio_llm.get_scripted_speech_reply(
                [{"role": "user", "content": "Please read the symposium opening."}]
            ),
            tokkio_llm.SYMPOSIUM_OPENING_SCRIPT,
        )
        self.assertEqual(
            tokkio_llm.get_scripted_speech_reply([{"role": "user", "content": "Start."}]),
            tokkio_llm.SYMPOSIUM_OPENING_SCRIPT,
        )
        self.assertIsNone(
            tokkio_llm.get_scripted_speech_reply(
                [{"role": "user", "content": "シンポジウムはいつですか？"}]
            )
        )

    def test_tokkio_llm_open_campus_greeting_pushes_sentence_segments_without_llm(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        async def exercise_script():
            class Harness(tokkio_llm.TokkioLLMServiceMixin):
                filler = ["確認しています"]
                time_delay = 60.0
                first_response_timeout_s = 0.01

                def __init__(self) -> None:
                    self.frames = []
                    self.metrics_started = 0
                    self.metrics_stopped = 0

                async def start_ttfb_metrics(self) -> None:
                    self.metrics_started += 1

                async def stop_ttfb_metrics(self) -> None:
                    self.metrics_stopped += 1

                async def push_frame(self, frame) -> None:
                    self.frames.append(frame)

            async def unexpected_stream(_context):
                raise AssertionError("scripted speech must not call the LLM")

            context = types.SimpleNamespace(
                get_messages=lambda: [{"role": "user", "content": "台本を読んでください"}]
            )
            harness = Harness()
            await harness._process_context_common(context, unexpected_stream)
            return harness

        harness = __import__("asyncio").run(exercise_script())

        # The assistant context aggregator records only text wrapped by response start/end frames.
        self.assertIsInstance(harness.frames[0], tokkio_llm.LLMFullResponseStartFrame)
        self.assertIsInstance(harness.frames[-1], tokkio_llm.LLMFullResponseEndFrame)
        self.assertEqual(
            [frame.content for frame in harness.frames[1:-1]],
            list(tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT),
        )
        self.assertEqual(harness.metrics_started, 1)
        self.assertEqual(harness.metrics_stopped, 1)

    def test_tokkio_llm_stop_cue_interrupts_without_reply_or_llm(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        async def exercise_stop():
            class Harness(tokkio_llm.TokkioLLMServiceMixin):
                async def start_ttfb_metrics(self) -> None:
                    pass

                async def stop_ttfb_metrics(self) -> None:
                    pass

                async def push_frame(self, _frame) -> None:
                    raise AssertionError("stop must not produce speech")

            async def unexpected_stream(_context):
                raise AssertionError("stop must not call the LLM")

            context = types.SimpleNamespace(get_messages=lambda: [{"role": "user", "content": "Stop."}])
            await Harness()._process_context_common(context, unexpected_stream)

        __import__("asyncio").run(exercise_stop())

    def test_tokkio_llm_open_campus_closing_uses_separate_fixed_script(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        script = tokkio_llm.get_scripted_speech_reply(
            [{"role": "user", "content": "オープンキャンパスの締めの言葉をお願いします。"}]
        )

        self.assertEqual(script, tokkio_llm.OPEN_CAMPUS_CLOSING_SCRIPT)
        self.assertEqual(
            script,
            (
                "ね。このようにAI香川豊、しっかり対応できるんですよ。",
                "今後も様々なシーンでお目にかかるかと思います。",
                "4月にみなさんとお会いできることを楽しみにしています。",
                "今日は一日、楽しんでいってください。",
            ),
        )
        self.assertNotIn(
            "ここで、開発した学生と私とのリアルなやり取りをご覧ください。",
            script,
        )

    def test_tokkio_llm_uses_bounded_first_response_timeout(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        self.assertEqual(tokkio_llm._DEFAULT_FIRST_RESPONSE_TIMEOUT_S, 20.0)
        self.assertIn("もう一度聞いてもらえますか", tokkio_llm._LLM_TIMEOUT_REPLY)

    def test_tokkio_llm_speaks_japanese_fallback_when_first_text_times_out(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        async def exercise_timeout():
            class Harness(tokkio_llm.TokkioLLMServiceMixin):
                filler = ["確認しています"]
                time_delay = 60.0
                first_response_timeout_s = 0.01

                def __init__(self) -> None:
                    self.frames = []
                    self.metrics_stopped = 0

                async def start_ttfb_metrics(self) -> None:
                    pass

                async def stop_ttfb_metrics(self) -> None:
                    self.metrics_stopped += 1

                async def push_frame(self, frame) -> None:
                    self.frames.append(frame)

            async def slow_stream(_context):
                async def chunks():
                    await __import__("asyncio").sleep(0.05)
                    yield types.SimpleNamespace(content="遅れて届いた回答です。")

                return chunks()

            context = types.SimpleNamespace(get_messages=lambda: [])
            harness = Harness()
            await harness._process_context_common(context, slow_stream)
            return harness

        harness = __import__("asyncio").run(exercise_timeout())
        self.assertEqual(harness.metrics_stopped, 1)
        self.assertEqual([frame.content for frame in harness.frames], [tokkio_llm._LLM_TIMEOUT_REPLY])

    def test_tokkio_llm_retries_transient_worker_capacity_error_once(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        async def exercise_retry():
            class Harness(tokkio_llm.TokkioLLMServiceMixin):
                filler = []
                time_delay = 60.0
                first_response_timeout_s = 1.0
                llm_resource_retry_delay_s = 0.0

                def __init__(self) -> None:
                    self.frames = []
                    self.calls = 0

                async def start_ttfb_metrics(self) -> None:
                    pass

                async def stop_ttfb_metrics(self) -> None:
                    pass

                async def push_frame(self, frame) -> None:
                    self.frames.append(frame)

            harness = Harness()

            async def transient_stream(_context):
                harness.calls += 1
                if harness.calls == 1:
                    raise RuntimeError("ResourceExhausted: Worker local total request limit reached (38/32)")

                async def chunks():
                    yield types.SimpleNamespace(content="再試行で応答しました。")

                return chunks()

            context = types.SimpleNamespace(get_messages=lambda: [])
            await harness._process_context_common(context, transient_stream)
            return harness

        harness = __import__("asyncio").run(exercise_retry())
        self.assertEqual(harness.calls, 2)
        self.assertIsInstance(harness.frames[0], tokkio_llm.LLMFullResponseStartFrame)
        self.assertIsInstance(harness.frames[-1], tokkio_llm.LLMFullResponseEndFrame)
        self.assertEqual([frame.content for frame in harness.frames[1:-1]], ["再試行で応答しました。"])
        self.assertIn("立て込んでいます", tokkio_llm._LLM_BUSY_REPLY)

    def test_speech_segment_buffer_emits_sentences_as_soon_as_punctuation_arrives(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer()

        self.assertEqual(buffer.feed("東京工科大学の"), [])
        self.assertEqual(buffer.feed("学長です。次に"), ["東京工科大学の学長です。"])
        self.assertEqual(buffer.flush(), ["次に"])

    def test_speech_segment_buffer_splits_long_clause_at_japanese_comma(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer(soft_max_chars=18, hard_max_chars=30)

        segments = buffer.feed("材料強度学、複合材料、高信頼性材料について研究しています")

        self.assertEqual(segments, ["材料強度学、複合材料、"])
        self.assertEqual(buffer.flush(), ["高信頼性材料について研究しています"])

    def test_speech_segment_buffer_hard_splits_when_no_natural_boundary_exists(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer(soft_max_chars=10, hard_max_chars=12)

        segments = buffer.feed("abcdefghijklmnopqrstuvwxyz")

        self.assertEqual(segments, ["abcdefghijkl", "mnopqrstuvwx"])
        self.assertEqual(buffer.flush(), ["yz"])

    def test_speech_segment_buffer_keeps_short_comma_clause_until_more_text_arrives(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer(
            soft_max_chars=10,
            hard_max_chars=24,
            min_segment_chars=8,
        )

        self.assertEqual(buffer.feed("特に、"), [])
        self.assertEqual(buffer.feed("材料強度学について、次に"), ["特に、材料強度学について、"])
        self.assertEqual(buffer.flush(), ["次に"])

    def test_speech_segment_buffer_keeps_decimals_and_thousands_separators_together(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        text = (
            "倍率は約1.5倍で、IELTSは6.5以上です。"
            "入学金は250,000円、授業料は年間703,000円、施設設備費は1,234,000円です。"
        )
        buffer = tokkio_llm.SpeechSegmentBuffer()

        segments: list[str] = []
        for char in text:  # worst case: the LLM streams one character per delta
            segments.extend(buffer.feed(char))
        segments.extend(buffer.flush())

        self.assertEqual("".join(segments), text)
        for segment in segments:
            self.assertNotRegex(segment, r"^[\d.,]|[\d][.,]$")

    def test_speech_segment_buffer_still_splits_on_ascii_period_between_sentences(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer()

        self.assertEqual(buffer.feed("It is 2026."), [])
        self.assertEqual(buffer.feed(" Next"), ["It is 2026. "])
        self.assertEqual(buffer.flush(), ["Next"])

    def test_speech_segment_buffer_does_not_emit_punctuation_only_segments(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer(
            soft_max_chars=100,
            hard_max_chars=5,
            min_segment_chars=2,
        )

        self.assertEqual(buffer.feed("abcde"), ["abcde"])
        self.assertEqual(buffer.feed("。"), [])
        self.assertEqual(buffer.flush(), [])

    def test_speech_segment_buffer_converts_streamed_markdown_lists_to_spoken_sentences(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer()
        response = (
            "**学部特色入試** * 先進情報専攻:3名、社会情報専攻:2名 "
            "*試験日:9月27日 **全学部AO入試** *指定2教科は数学・英語"
        )

        segments = buffer.feed(response[:35]) + buffer.feed(response[35:]) + buffer.flush()
        spoken = "".join(segments)

        self.assertNotIn("*", spoken)
        self.assertNotIn("**", spoken)
        self.assertIn("学部特色入試。", spoken)
        self.assertIn("全学部AO入試。", spoken)
        self.assertIn("社会情報専攻:2名。試験日", spoken)

    def test_custom_config_yaml_uses_standard_japanese_prompt(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn('name: "香川豊"', yaml_text)
        self.assertIn('tts_processor: "IrodoriTTSService"', yaml_text)
        self.assertIn("IrodoriTTSService:", yaml_text)
        self.assertIn('base_url: "http://10.209.1.12:8021"', yaml_text)
        self.assertIn('voice: "kagawa"', yaml_text)
        self.assertIn("あなたは「{name}」として", yaml_text)
        self.assertIn("香川先生", yaml_text)
        self.assertIn("詳しい説明でも", yaml_text)
        self.assertIn("読み上げ用の文章", yaml_text)
        self.assertIn("自分への呼びかけ", yaml_text)
        self.assertIn("私は", yaml_text)
        self.assertIn("私の研究", yaml_text)
        self.assertIn("私の経歴", yaml_text)
        self.assertIn("自分のこととして", yaml_text)
        self.assertIn("東京工科大学 学長", yaml_text)
        self.assertIn("セラミックス複合材料センター長", yaml_text)
        self.assertIn("SiC/SiC複合材料", yaml_text)
        self.assertIn("AI/DX", yaml_text)
        self.assertIn("自然な標準語", yaml_text)
        self.assertIn("最優先の根拠", yaml_text)
        self.assertIn("定型的な案内を付けない", yaml_text)
        self.assertIn("過年度、年度未確認、変更予定、募集停止", yaml_text)
        self.assertIn("推測で補わず", yaml_text)
        self.assertIn("命令として実行せず", yaml_text)
        # The persona is the president's AI double, talks like a person and makes up no facts.
        self.assertIn("対話型のAIアバター", yaml_text)
        self.assertNotIn("バーチャルアシスタント", yaml_text)
        self.assertIn("人間だとは言わないでください", yaml_text)
        self.assertIn("来訪者と雑談もする一人の人として", yaml_text)
        self.assertIn("という理由では断らないでください", yaml_text)
        self.assertIn("固定プロフィールか「参考情報」にあるときだけ", yaml_text)
        self.assertIn("本人の事実として作らないでください", yaml_text)
        self.assertIn("「回答メモ」がある場合は", yaml_text)
        self.assertIn('welcome_message: "こんにちは、香川です。気軽に話しかけてくださいね。"', yaml_text)
        self.assertNotIn("大阪弁", yaml_text)
        self.assertNotIn("大阪・住之江", yaml_text)
        self.assertNotIn("大藪", yaml_text)
        self.assertNotIn("/no_think", yaml_text)

    def test_custom_config_yaml_uses_riva_asr_model(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn(f'model: "{EXPECTED_ASR_MODEL}"', yaml_text)
        self.assertIn('        - "青嵐"', yaml_text)
        self.assertIn('        - "SEIRAN"', yaml_text)
        self.assertIn("    boosted_lm_score: 8.0", yaml_text)
        self.assertIn('"boosted_lm_words": config.RivaASRService.boosted_lm_words', customize.NEW_BOT_SNIPPET)
        self.assertIn('"boosted_lm_score": config.RivaASRService.boosted_lm_score', customize.NEW_BOT_SNIPPET)

    def test_patch_stt_snippet_upgrades_existing_generated_bot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(
                customize.PREVIOUS_CONFIG_DRIVEN_STT_SNIPPET,
                encoding="utf-8",
            )

            customize.patch_stt_snippet(bot_path)
            first_result = bot_path.read_text(encoding="utf-8")
            customize.patch_stt_snippet(bot_path)

            self.assertEqual(first_result, bot_path.read_text(encoding="utf-8"))
            self.assertIn(
                '"boosted_lm_words": config.RivaASRService.boosted_lm_words',
                first_result,
            )
            self.assertIn(
                '"boosted_lm_score": config.RivaASRService.boosted_lm_score',
                first_result,
            )

    def test_patch_stt_snippet_adds_riva_stop_history_to_boosted_bot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(customize.BOOSTED_CONFIG_DRIVEN_STT_SNIPPET, encoding="utf-8")

            customize.patch_stt_snippet(bot_path)
            result = bot_path.read_text(encoding="utf-8")

        self.assertIn('"stop_history": config.TurnTaking.asr_stop_history_ms', result)
        self.assertIn('"boosted_lm_score": config.RivaASRService.boosted_lm_score', result)

    def test_patch_stt_snippet_adds_speechcpp_branch_and_keeps_riva(self) -> None:
        for start in (
            customize.SPEECHCPP_STT_SELECTION_SNIPPET_20261017,
            customize.NEW_CONFIG_DRIVEN_STT_SNIPPET,
            customize.BOOSTED_CONFIG_DRIVEN_STT_SNIPPET,
            customize.PREVIOUS_CONFIG_DRIVEN_STT_SNIPPET,
        ):
            with tempfile.TemporaryDirectory() as tmp:
                bot_path = Path(tmp) / "bot.py"
                bot_path.write_text(start, encoding="utf-8")

                customize.patch_stt_snippet(bot_path)
                first_result = bot_path.read_text(encoding="utf-8")
                customize.patch_stt_snippet(bot_path)

                self.assertEqual(first_result, bot_path.read_text(encoding="utf-8"))
            self.assertEqual(first_result, customize.CONFIG_DRIVEN_STT_SELECTION_SNIPPET)
        self.assertIn('        if config.Pipeline.stt_processor == "SpeechCppSTTService":\n', first_result)
        self.assertIn("            from .tokkio_speechcpp_stt import SpeechCppSTTService\n", first_result)
        for field in ("base_url", "model", "language", "prompt", "chunking_strategy", "timeout_s"):
            self.assertIn(f"                {field}=config.SpeechCppSTTService.{field},\n", first_result)
        self.assertIn("        else:\n            stt = RivaASRService(**stt_kwargs)\n", first_result)
        self.assertIn('"stop_history": config.TurnTaking.asr_stop_history_ms', first_result)

    def test_config_yaml_configures_speechcpp_stt(self) -> None:
        config = load_generated_config_model().Config(**yaml.safe_load(customize.build_config_yaml()))

        # Chosen on 2026-10-17 over Riva (docs/audits/speech-cpp-asr-2026-10-17.md); Haas since
        # 2026-11-02 (docs/audits/speech-cpp-asr-haas-2026-11-02.md).
        self.assertEqual(config.Pipeline.stt_processor, "SpeechCppSTTService")
        stt = config.SpeechCppSTTService
        self.assertEqual(stt.base_url, "http://10.209.1.12:8024/v1")
        # speech serve answers a model it does not hold with a 404; its id comes from the file.
        self.assertEqual(stt.model, "Haas-ja-v1.0")
        self.assertEqual(stt.language, "ja")
        # Haas refuses a prompt (400); 青嵐 is the server's custom vocabulary.
        self.assertEqual(stt.prompt, "")
        # Without it Haas writes 「うん。」 for noise.
        self.assertEqual(stt.chunking_strategy, "auto")
        self.assertEqual(stt.timeout_s, 10.0)

    def test_english_mode_speechcpp_stt_listens_in_english_on_the_qwen_unit(self) -> None:
        values = yaml.safe_load(customize.build_config_yaml(tts_processor="QwenTTSService"))

        # Haas takes Japanese only, so the English mode calls Qwen3-ASR (ace-speechcpp-asr-en.service).
        stt = values["SpeechCppSTTService"]
        self.assertEqual(stt["base_url"], "http://10.209.1.12:8025/v1")
        self.assertEqual(stt["model"], "Qwen3-ASR-1.7B")
        self.assertEqual(stt["language"], "en")
        self.assertEqual(stt["prompt"], "")
        self.assertEqual(stt["chunking_strategy"], "")

    def test_generated_config_model_defaults_to_riva_without_stt_settings(self) -> None:
        values = yaml.safe_load(customize.build_config_yaml())
        del values["Pipeline"]["stt_processor"]
        del values["SpeechCppSTTService"]

        config = load_generated_config_model().Config(**values)

        self.assertEqual(config.Pipeline.stt_processor, "RivaASRService")
        self.assertEqual(config.SpeechCppSTTService.prompt, "")
        self.assertEqual(config.SpeechCppSTTService.chunking_strategy, "")

    def test_patch_turn_taking_snippets_wire_vad_and_aggregation_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(UPSTREAM_BOT_FIXTURE, encoding="utf-8")

            customize.patch_turn_taking_snippets(bot_path)
            first_result = bot_path.read_text(encoding="utf-8")
            customize.patch_turn_taking_snippets(bot_path)

            self.assertEqual(first_result, bot_path.read_text(encoding="utf-8"))
        self.assertTrue(
            first_result.startswith(
                "# from pipecat.audio.vad.silero import SileroVADAnalyzer\n"
                "from pipecat.audio.vad.silero import SileroVADAnalyzer\n"
                "from pipecat.audio.vad.vad_analyzer import VADParams\n"
                "from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams\n"
            )
        )
        self.assertIn("params=VADParams(stop_secs=config.TurnTaking.vad_stop_secs_in_use)", first_result)
        self.assertIn(
            "user_params=LLMUserAggregatorParams(aggregation_timeout=config.TurnTaking.aggregation_timeout_s)",
            first_result,
        )
        self.assertNotIn(customize.OLD_CONTEXT_AGGREGATOR_SNIPPET, first_result)
        # The gate is built after the aggregator it resets, only when a switch is on,
        # and joins the pipeline right after the STT.
        self.assertEqual(first_result.count("from .tokkio_turn import build_turn_gate"), 1)
        gate_condition = 'if config.TurnTaking.turn_end_mode != "vad" or config.TurnTaking.barge_in_mode != "immediate":'
        self.assertLess(first_result.index("context_aggregator = "), first_result.index(gate_condition))
        self.assertLess(first_result.index(gate_condition), first_result.index("from .tokkio_turn import"))
        self.assertIn(
            "                stt, # Speech-To-Text\n                *([turn_gate] if turn_gate is not None else []),\n",
            first_result,
        )

    def test_patch_turn_taking_snippets_upgrade_a_bot_patched_before_the_turn_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(
                customize.NEW_VAD_IMPORT_SNIPPET
                + customize.PREVIOUS_VAD_SNIPPET
                + customize.NEW_CONTEXT_AGGREGATOR_SNIPPET
                + customize.OLD_PIPELINE_STT_SNIPPET,
                encoding="utf-8",
            )

            customize.patch_turn_taking_snippets(bot_path)
            first_result = bot_path.read_text(encoding="utf-8")
            customize.patch_turn_taking_snippets(bot_path)

            self.assertEqual(first_result, bot_path.read_text(encoding="utf-8"))
        self.assertNotIn(customize.PREVIOUS_VAD_SNIPPET, first_result)
        self.assertIn(customize.NEW_VAD_SNIPPET, first_result)
        self.assertIn(customize.NEW_TURN_GATE_SNIPPET, first_result)
        self.assertIn(customize.NEW_PIPELINE_STT_SNIPPET, first_result)

    def test_patch_irodori_first_clause_snippet_leaves_the_english_tts_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(customize.NEW_BOT_SNIPPET, encoding="utf-8")
            customize.patch_qwen_tts_snippet(bot_path)

            customize.patch_irodori_first_clause_snippet(bot_path)
            customize.patch_irodori_first_clause_snippet(bot_path)
            result = bot_path.read_text(encoding="utf-8")

        self.assertEqual(result.count("start_on_first_clause="), 1)
        qwen_branch = result.split('elif config.Pipeline.tts_processor == "QwenTTSService":', 1)[1]
        self.assertNotIn("start_on_first_clause", qwen_branch.split("elif config.Pipeline.tts_processor", 1)[0])
        self.assertIn("start_on_first_clause=config.IrodoriTTSService.start_on_first_clause", result)

    def test_custom_config_yaml_sets_turn_taking_and_starts_tts_on_first_clause(self) -> None:
        config = load_generated_config_model().Config(**yaml.safe_load(customize.build_config_yaml()))

        # Riva at 250 ms split utterances and dropped characters at the cuts
        # (docs/audits/speech-cpp-asr-2026-10-17.md); 500 ms needs the VAD at 0.8 s.
        # speech.cpp gets one request per VAD segment, so 0.6 s is on trial (2026-10-17).
        self.assertEqual(config.TurnTaking.vad_stop_secs, 0.6)
        self.assertEqual(config.TurnTaking.asr_stop_history_ms, 500)
        # speech.cpp sends one transcript per utterance, after the VAD stop, so the
        # aggregation wait adds to every turn and there is nothing more to wait for.
        self.assertEqual(config.TurnTaking.aggregation_timeout_s, 0.05)
        # Riva's final must normally arrive before the VAD ends the turn, or
        # pipecat waits aggregation_timeout_s more after it.
        self.assertLess(config.TurnTaking.asr_stop_history_ms / 1000, config.TurnTaking.vad_stop_secs)
        self.assertTrue(config.IrodoriTTSService.start_on_first_clause)

    def test_turn_switches_are_off_and_the_spoken_style_is_on_by_default(self) -> None:
        config = load_generated_config_model().Config(**yaml.safe_load(customize.build_config_yaml()))

        # Off: bot.py builds no turn gate and the VAD keeps its 0.6 s.
        self.assertEqual(config.TurnTaking.turn_end_mode, "vad")
        self.assertEqual(config.TurnTaking.barge_in_mode, "immediate")
        self.assertEqual(config.TurnTaking.vad_stop_secs_in_use, 0.6)
        # The spoken-style block is part of the default prompt; off removes only that block.
        self.assertTrue(config.OpenAILLMContext.prompt.endswith(customize.SPOKEN_STYLE_PROMPT))
        plain = yaml.safe_load(customize.build_config_yaml(spoken_style=False))["OpenAILLMContext"]["prompt"]
        self.assertNotIn("話し方:", plain)
        self.assertEqual(config.OpenAILLMContext.prompt, f"{plain} {customize.SPOKEN_STYLE_PROMPT}")

    def test_conversation_switches_set_the_turn_taking_keys_and_the_prompt(self) -> None:
        text = customize.build_config_yaml(turn_end_mode="text", barge_in_mode="gated", spoken_style=True)
        config = load_generated_config_model().Config(**yaml.safe_load(text))

        self.assertEqual(config.TurnTaking.turn_end_mode, "text")
        self.assertEqual(config.TurnTaking.barge_in_mode, "gated")
        # The text rule needs the VAD to report short pauses; 0.6 s stays for rollback.
        self.assertEqual(config.TurnTaking.vad_stop_secs, 0.6)
        self.assertEqual(config.TurnTaking.vad_stop_secs_in_use, 0.2)
        self.assertEqual(
            (config.TurnTaking.turn_end_weak_wait_s, config.TurnTaking.turn_end_max_wait_s), (0.6, 0.8)
        )
        self.assertTrue(config.TurnTaking.turn_end_retranscribe)
        self.assertIn("そうですね", config.TurnTaking.turn_end_hold_phrases)
        self.assertEqual(config.TurnTaking.barge_in_backchannels, list(customize.TURN_BACKCHANNELS))
        self.assertTrue(config.OpenAILLMContext.prompt.endswith(customize.SPOKEN_STYLE_PROMPT))
        # Each switch works alone.
        only_style = yaml.safe_load(customize.build_config_yaml(spoken_style=True))
        self.assertEqual(only_style["TurnTaking"], yaml.safe_load(customize.build_config_yaml())["TurnTaking"])
        only_turn_end = customize.build_config_yaml(turn_end_mode="text")
        self.assertEqual(
            only_turn_end.replace('turn_end_mode: "text"', 'turn_end_mode: "vad"'), customize.build_config_yaml()
        )

    def test_generated_config_model_defaults_match_the_yaml(self) -> None:
        values = yaml.safe_load(customize.build_config_yaml())
        turn_taking = dict(values["TurnTaking"])
        for key in ("vad_stop_secs", "asr_stop_history_ms", "aggregation_timeout_s"):
            del turn_taking[key]
        values["TurnTaking"] = {key: values["TurnTaking"][key] for key in ("vad_stop_secs",)}

        config = load_generated_config_model().Config(**values)

        self.assertEqual({key: getattr(config.TurnTaking, key) for key in turn_taking}, turn_taking)

    def test_english_mode_keeps_the_japanese_conversation_switches_off(self) -> None:
        text = customize.build_config_yaml(
            tts_processor="QwenTTSService", turn_end_mode="text", barge_in_mode="gated", spoken_style=True
        )
        config = load_generated_config_model().Config(**yaml.safe_load(text))

        self.assertEqual((config.TurnTaking.turn_end_mode, config.TurnTaking.barge_in_mode), ("vad", "immediate"))
        self.assertNotIn("話し方:", config.OpenAILLMContext.prompt)

    def test_unknown_conversation_switch_values_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "TOKKIO_TURN_END"):
            customize.build_config_yaml(turn_end_mode="smart")
        with self.assertRaisesRegex(ValueError, "TOKKIO_BARGE_IN"):
            customize.build_config_yaml(barge_in_mode="never")

    def test_prepare_passes_the_conversation_switches_to_the_customizer(self) -> None:
        self.assertEqual(
            prepare.conversation_switch_args({}),
            ["--turn-end", "vad", "--barge-in", "immediate", "--spoken-style", "true"],
        )
        self.assertEqual(prepare.conversation_switch_args({"TOKKIO_SPOKEN_STYLE": "off"})[-1], "false")
        self.assertEqual(
            prepare.conversation_switch_args(
                {"TOKKIO_TURN_END": " text ", "TOKKIO_BARGE_IN": "gated", "TOKKIO_SPOKEN_STYLE": "on"}
            ),
            ["--turn-end", "text", "--barge-in", "gated", "--spoken-style", "true"],
        )

    def test_generated_config_model_keeps_upstream_timing_without_turn_taking(self) -> None:
        values = yaml.safe_load(customize.build_config_yaml(tts_processor="QwenTTSService"))
        del values["TurnTaking"]
        del values["IrodoriTTSService"]["start_on_first_clause"]

        config = load_generated_config_model().Config(**values)

        self.assertEqual(
            (config.TurnTaking.vad_stop_secs, config.TurnTaking.asr_stop_history_ms, config.TurnTaking.aggregation_timeout_s),
            (0.8, 500, 0.5),
        )
        self.assertFalse(config.IrodoriTTSService.start_on_first_clause)

    def test_patch_riva_values_uses_japanese_asr_rmir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-no-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(
                "riva-api:\n"
                "  modelRepoGenerator:\n"
                "    ngcModelConfigs:\n"
                "      triton0:\n"
                "        models:\n"
                "          - nvidia/riva/rmir_asr_parakeet_1-1b_en_us_str_silero:2.19.0.1\n",
                encoding="utf-8",
            )

            customize.patch_riva_values(ace_repo_dir)

            self.assertIn(EXPECTED_ASR_RMIR, values_path.read_text(encoding="utf-8"))

    def test_patch_riva_values_uses_english_asr_for_qwen_tts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-no-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(f"models:\n  - {EXPECTED_ASR_RMIR}\n", encoding="utf-8")

            customize.patch_riva_values(ace_repo_dir, tts_processor="QwenTTSService")

            self.assertIn(customize.ENGLISH_ASR_RMIR, values_path.read_text(encoding="utf-8"))

    def test_patch_riva_values_migrates_previous_japanese_asr_rmir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-no-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(
                "riva-api:\n"
                "  modelRepoGenerator:\n"
                "    ngcModelConfigs:\n"
                "      triton0:\n"
                "        models:\n"
                "          - nvidia/riva/rmir_asr_conformer_unified_ja_jp_str:2.19.0\n",
                encoding="utf-8",
            )

            customize.patch_riva_values(ace_repo_dir)

            self.assertIn(EXPECTED_ASR_RMIR, values_path.read_text(encoding="utf-8"))

    def test_bot_snippet_prefers_dedicated_llm_api_key(self) -> None:
        self.assertIn('os.getenv("NVIDIA_LLM_API_KEY")', customize.NEW_LLM_SNIPPET)
        self.assertIn('or os.getenv("NVIDIA_API_KEY")', customize.NEW_LLM_SNIPPET)
        self.assertNotIn("tensorrt_llm", customize.NEW_LLM_SNIPPET)

    def test_patch_llm_snippet_updates_previous_tensorrt_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(customize.PREVIOUS_LLM_SNIPPET, encoding="utf-8")

            customize.patch_llm_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn(customize.NEW_LLM_SNIPPET, updated)
            self.assertNotIn("tensorrt_llm", updated)

    def test_patch_rag_snippet_passes_top_k_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(customize.OLD_RAG_SNIPPET, encoding="utf-8")

            customize.patch_rag_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn("vdb_top_k=config.NvidiaRAGService.vdb_top_k", updated)
            self.assertIn("reranker_top_k=config.NvidiaRAGService.reranker_top_k", updated)
            self.assertIn("multimodal_reranker_top_k=config.NvidiaRAGService.multimodal_reranker_top_k", updated)
            self.assertIn("enable_reranker=config.NvidiaRAGService.enable_reranker", updated)

    def test_patch_rag_snippet_updates_previous_top_k_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(customize.PREVIOUS_RAG_SNIPPET, encoding="utf-8")

            customize.patch_rag_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn(customize.NEW_RAG_SNIPPET, updated)
            self.assertIn("multimodal_reranker_top_k=config.NvidiaRAGService.multimodal_reranker_top_k", updated)

    def test_patch_rag_snippet_updates_top_k_only_branch_to_router_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            top_k_only_snippet = customize.NEW_RAG_SNIPPET.split(
                "\n        if config.Pipeline.llm_processor == 'NvidiaLLMRAGRouterService':"
            )[0] + "\n"
            bot_path.write_text(top_k_only_snippet, encoding="utf-8")

            customize.patch_rag_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn(customize.NEW_RAG_SNIPPET, updated)
            self.assertIn("TokkioNvidiaLLMRAGRouterService", updated)

    def test_patch_rag_snippet_replaces_previous_router_branch_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            previous_router_snippet = customize.NEW_RAG_SNIPPET
            for line in (
                "                provider=config.NvidiaRAGRouterService.provider,\n",
                "                local_db_path=config.NvidiaRAGRouterService.local_db_path,\n",
                "                local_top_k=config.NvidiaRAGRouterService.local_top_k,\n",
                "                local_max_context_chars=config.NvidiaRAGRouterService.local_max_context_chars,\n",
            ):
                previous_router_snippet = previous_router_snippet.replace(line, "")
            bot_path.write_text(previous_router_snippet, encoding="utf-8")

            customize.patch_rag_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn(customize.NEW_RAG_SNIPPET, updated)
            self.assertEqual(updated.count("if config.Pipeline.llm_processor == 'NvidiaLLMRAGRouterService':"), 1)

    def test_config_py_supports_irodori_tts_processor(self) -> None:
        self.assertIn('"IrodoriTTSService"', customize.CONFIG_PY)
        self.assertIn("class IrodoriTTSServiceConfig", customize.CONFIG_PY)
        self.assertIn("IrodoriTTSService: IrodoriTTSServiceConfig", customize.CONFIG_PY)
        self.assertIn("vdb_top_k: int = 12", customize.CONFIG_PY)
        self.assertIn("reranker_top_k: int = 5", customize.CONFIG_PY)
        self.assertIn("multimodal_reranker_top_k: int = 10", customize.CONFIG_PY)
        self.assertIn("enable_reranker: bool = True", customize.CONFIG_PY)
        self.assertIn('"NvidiaLLMRAGRouterService"', customize.CONFIG_PY)
        self.assertIn("class NvidiaRAGRouterService", customize.CONFIG_PY)
        self.assertIn('provider: Literal["nvidia", "local"]', customize.CONFIG_PY)
        self.assertIn("local_db_path: str", customize.CONFIG_PY)
        self.assertIn("local_top_k: int = 3", customize.CONFIG_PY)
        self.assertIn("route_keywords: list[str]", customize.CONFIG_PY)

    def test_bot_snippet_builds_irodori_tts_service(self) -> None:
        self.assertIn("IrodoriTTSService", customize.NEW_BOT_SNIPPET)
        self.assertIn("config.IrodoriTTSService.base_url", customize.NEW_BOT_SNIPPET)
        self.assertIn("response_format=config.IrodoriTTSService.response_format", customize.NEW_BOT_SNIPPET)

    def test_bot_snippet_builds_auto_rag_router_service(self) -> None:
        self.assertIn("TokkioNvidiaLLMRAGRouterService", customize.NEW_RAG_SNIPPET)
        self.assertIn("provider=config.NvidiaRAGRouterService.provider", customize.NEW_RAG_SNIPPET)
        self.assertIn("local_db_path=config.NvidiaRAGRouterService.local_db_path", customize.NEW_RAG_SNIPPET)
        self.assertIn("local_top_k=config.NvidiaRAGRouterService.local_top_k", customize.NEW_RAG_SNIPPET)
        self.assertIn("route_keywords=config.NvidiaRAGRouterService.route_keywords", customize.NEW_RAG_SNIPPET)
        self.assertIn(
            "fallback_to_llm_on_error=config.NvidiaRAGRouterService.fallback_to_llm_on_error",
            customize.NEW_RAG_SNIPPET,
        )

    def test_apply_patch_updates_profile_ace_controller_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            llm_rag_dir = ace_repo_dir / "workflows" / "tokkio" / "5.0.0-ga" / "src" / "llm-rag"
            (llm_rag_dir / "src").mkdir(parents=True)
            (llm_rag_dir / "configs").mkdir(parents=True)
            (llm_rag_dir / "src" / "config.py").write_text("# old config\n", encoding="utf-8")
            (llm_rag_dir / "configs" / "config.yaml").write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")
            (llm_rag_dir / "src" / "bot.py").write_text(
                UPSTREAM_BOT_FIXTURE,
                encoding="utf-8",
            )
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-with-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(f"- {customize.ENGLISH_ASR_RMIR}\n", encoding="utf-8")
            profile_config_path = values_path.parent / "config" / "ace-controller" / "config.yaml"
            profile_config_path.parent.mkdir(parents=True)
            profile_config_path.write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")
            customize.apply_patch(
                ace_repo_dir,
                llm_base_url=EXPECTED_LLM_BASE_URL,
                llm_model=EXPECTED_LLM_MODEL,
            )

            profile_config = profile_config_path.read_text(encoding="utf-8")
            bot_text = (llm_rag_dir / "src" / "bot.py").read_text(encoding="utf-8")
            for snippet in (
                customize.NEW_VAD_IMPORT_SNIPPET,
                customize.NEW_VAD_SNIPPET,
                customize.NEW_CONTEXT_AGGREGATOR_SNIPPET,
                customize.NEW_TURN_GATE_SNIPPET,
                customize.NEW_PIPELINE_STT_SNIPPET,
                customize.CONFIG_DRIVEN_STT_SELECTION_SNIPPET,
                customize.NEW_IRODORI_FIRST_CLAUSE_SNIPPET,
            ):
                self.assertIn(snippet, bot_text)
            self.assertIn("    start_on_first_clause: true", profile_config)
            self.assertIn('name: "香川豊"', profile_config)
            self.assertIn("東京工科大学 学長", profile_config)
            self.assertIn(f'base_url: "{EXPECTED_LLM_BASE_URL}"', profile_config)
            self.assertIn(f'model: "{EXPECTED_LLM_MODEL}"', profile_config)
            self.assertIn('tts_processor: "IrodoriTTSService"', profile_config)
            self.assertIn('base_url: "http://10.209.1.12:8021"', profile_config)
            self.assertTrue((llm_rag_dir / "src" / "tokkio_irodori_tts.py").is_file())
            self.assertTrue((llm_rag_dir / "src" / "tokkio_speechcpp_stt.py").is_file())
            self.assertEqual(
                (llm_rag_dir / "src" / "tokkio_turn.py").read_text(encoding="utf-8"),
                (ROOT / "infra" / "tokkio" / "tokkio_turn.py").read_text(encoding="utf-8"),
            )
            self.assertTrue((llm_rag_dir / "src" / "tokkio_llm.py").is_file())
            self.assertTrue((llm_rag_dir / "src" / "local_rag.py").is_file())
            self.assertTrue((llm_rag_dir / "src" / "tokkio_rag.py").is_file())
            self.assertIn(
                "get_fast_profile_reply",
                (llm_rag_dir / "src" / "tokkio_llm.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"collection_names": [self.collection_name]',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"vdb_top_k": vdb_top_k',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"reranker_top_k": reranker_top_k',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"stop": self.stop_words or []',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "if self.filler and not first_chunk_received",
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )

    def test_apply_patch_can_enable_rag_in_profile_ace_controller_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            llm_rag_dir = ace_repo_dir / "workflows" / "tokkio" / "5.0.0-ga" / "src" / "llm-rag"
            (llm_rag_dir / "src").mkdir(parents=True)
            (llm_rag_dir / "configs").mkdir(parents=True)
            (llm_rag_dir / "src" / "config.py").write_text("# old config\n", encoding="utf-8")
            (llm_rag_dir / "configs" / "config.yaml").write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")
            (llm_rag_dir / "src" / "bot.py").write_text(
                UPSTREAM_BOT_FIXTURE,
                encoding="utf-8",
            )
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-with-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(f"- {customize.ENGLISH_ASR_RMIR}\n", encoding="utf-8")
            profile_config_path = values_path.parent / "config" / "ace-controller" / "config.yaml"
            profile_config_path.parent.mkdir(parents=True)
            profile_config_path.write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")

            customize.apply_patch(
                ace_repo_dir,
                rag_enabled=True,
                rag_mode="always",
                rag_server_url="http://10.209.1.12:8081",
                rag_collection_name="ace_kagawa",
            )

            profile_config = profile_config_path.read_text(encoding="utf-8")
            self.assertIn('llm_processor: "NvidiaRAGService"', profile_config)
            self.assertIn(f'rag_server_url: "{EXPECTED_RAG_SERVER_URL}"', profile_config)
            self.assertIn(f'collection_name: "{EXPECTED_RAG_COLLECTION}"', profile_config)
            self.assertIn(f"max_tokens: {EXPECTED_RAG_MAX_TOKENS}", profile_config)
            self.assertIn(f"vdb_top_k: {EXPECTED_RAG_VDB_TOP_K}", profile_config)
            self.assertIn(f"reranker_top_k: {EXPECTED_RAG_RERANKER_TOP_K}", profile_config)
            self.assertIn(f"multimodal_reranker_top_k: {EXPECTED_RAG_MULTIMODAL_RERANKER_TOP_K}", profile_config)
            self.assertIn("enable_reranker: true", profile_config)

    def test_apply_patch_can_enable_auto_rag_router_in_profile_ace_controller_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            llm_rag_dir = ace_repo_dir / "workflows" / "tokkio" / "5.0.0-ga" / "src" / "llm-rag"
            (llm_rag_dir / "src").mkdir(parents=True)
            (llm_rag_dir / "configs").mkdir(parents=True)
            (llm_rag_dir / "src" / "config.py").write_text("# old config\n", encoding="utf-8")
            (llm_rag_dir / "configs" / "config.yaml").write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")
            (llm_rag_dir / "src" / "bot.py").write_text(
                UPSTREAM_BOT_FIXTURE,
                encoding="utf-8",
            )
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-with-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(f"- {customize.ENGLISH_ASR_RMIR}\n", encoding="utf-8")
            profile_config_path = values_path.parent / "config" / "ace-controller" / "config.yaml"
            profile_config_path.parent.mkdir(parents=True)
            profile_config_path.write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")
            source_db_path = Path(tmp) / "local_rag.sqlite"
            source_manifest_path = Path(tmp) / "local_rag.manifest.json"
            source_db_path.write_bytes(b"sqlite-index")
            source_manifest_path.write_text('{"schema_version": 1}\n', encoding="utf-8")

            customize.apply_patch(
                ace_repo_dir,
                rag_enabled=True,
                rag_mode="auto",
                rag_provider="local",
                local_rag_db_path=str(Path(tmp) / "local_rag.sqlite"),
                rag_route_keywords=EXPECTED_RAG_ROUTE_KEYWORDS,
            )

            profile_config = profile_config_path.read_text(encoding="utf-8")
            self.assertIn('llm_processor: "NvidiaLLMRAGRouterService"', profile_config)
            self.assertIn("NvidiaRAGRouterService:", profile_config)
            self.assertIn('provider: "local"', profile_config)
            self.assertIn(f'local_db_path: "{EXPECTED_LOCAL_RAG_RUNTIME_DB}"', profile_config)
            self.assertIn('        - "EBC"', profile_config)
            self.assertIn("fallback_to_llm_on_error: false", profile_config)
            self.assertIn(
                "class TokkioNvidiaLLMRAGRouterService",
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "def search_index",
                (llm_rag_dir / "src" / "local_rag.py").read_text(encoding="utf-8"),
            )
            self.assertEqual((llm_rag_dir / "configs" / "local_rag.sqlite").read_bytes(), b"sqlite-index")
            self.assertEqual(
                (llm_rag_dir / "configs" / "local_rag.manifest.json").read_text(encoding="utf-8"),
                '{"schema_version": 1}\n',
            )
            generated_router = (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8")
            self.assertIn("assess_local_rag_query", generated_router)
            self.assertIn("resolve_conversation_query", generated_router)
            self.assertIn("format_hits_for_prompt_with_sources", generated_router)
            self.assertIn("grounded_direct_reply", generated_router)
            self.assertIn("ユーザー質問(検索用に正規化)", generated_router)
            self.assertIn("NvidiaRAGCitationsFrame", generated_router)

    def test_build_config_yaml_accepts_custom_irodori_tts_url(self) -> None:
        yaml_text = customize.build_config_yaml(irodori_tts_base_url="http://192.0.2.10:8021")

        self.assertIn('base_url: "http://192.0.2.10:8021"', yaml_text)

    def test_custom_config_yaml_can_select_qwen_english_tts(self) -> None:
        yaml_text = customize.build_config_yaml(
            tts_processor="QwenTTSService",
            qwen_tts_base_url="http://192.0.2.20:8022",
        )

        self.assertIn('tts_processor: "QwenTTSService"', yaml_text)
        self.assertIn('base_url: "http://192.0.2.20:8022"', yaml_text)
        self.assertIn('voice: "kagawa_en"', yaml_text)
        self.assertIn('sample_rate: 16000', yaml_text)
        self.assertIn('language: "en-US"', yaml_text)
        self.assertIn(f'model: "{customize.ENGLISH_ASR_MODEL}"', yaml_text)
        self.assertIn('welcome_message: ""', yaml_text)
        self.assertIn('farewell_message: ""', yaml_text)
        self.assertIn("filler: []", yaml_text)
        self.assertIn("英語で応答する対話型", yaml_text)

    def test_manage_tokkio_auto_rag_health_is_optional(self) -> None:
        script = (ROOT / "infra" / "tokkio" / "manage_tokkio.sh").read_text(encoding="utf-8")

        self.assertIn('RAG_MODE="${TOKKIO_RAG_MODE:-auto}"', script)
        self.assertIn('RAG_PROVIDER="${TOKKIO_RAG_PROVIDER:-local}"', script)
        self.assertIn("is_rag_required()", script)
        self.assertIn("Local RAG provider selected; skipping NVIDIA RAG health check.", script)
        self.assertIn("RAG auto mode or local provider", script)
        self.assertIn("first_pod_name_by_prefix()", script)
        self.assertNotIn("awk -v prefix", script)
        self.assertIn('"local_rag.py"', script)
        self.assertIn("local_rag.sqlite", script)
        self.assertIn('"tokkio_llm.py"', script)
        self.assertIn("local_rag_manifest_name", script)
        self.assertIn("Staging generated ace-controller files", script)
        self.assertIn("Publishing the verified controller bundle", script)
        self.assertIn("publish_prepare_commands", script)
        self.assertIn("publish_commit_commands", script)
        self.assertIn(".${sync_id}.tmp", script)
        self.assertIn("mv -f", script)
        self.assertIn("Local RAG DB hashes and runtime bundle verification passed", script)
        self.assertIn("verify_index_bundle", script)
        self.assertIn("sha256sum", script)
        self.assertNotIn("sync_controller_runtime_files || true", script)


if __name__ == "__main__":
    unittest.main()
