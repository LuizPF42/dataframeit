"""adataframeit: o mesmo resultado de dataframeit, sem threads e sem time.sleep."""

import asyncio
import inspect
import linecache
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

import dataframeit as pacote
from dataframeit import (
    ProviderAbortError,
    ProviderError,
    ProviderOverloadedError,
    ProviderTransientError,
    RowResult,
    adataframeit,
    core,
    dataframeit,
    errors,
)


class Modelo(BaseModel):
    campo1: str
    campo2: str


def _resposta(texto, uso=None):
    """O que o structured output do LangChain devolve com include_raw=True."""
    raw = SimpleNamespace(usage_metadata=uso) if uso else None
    return {
        "raw": raw,
        "parsed": Modelo(campo1=texto, campo2=texto.upper()),
        "parsing_error": None,
    }


def _texto(mensagens):
    """O texto da linha: com o prompt "{texto}", a primeira mensagem é o próprio texto."""
    return mensagens if isinstance(mensagens, str) else mensagens[0][1].split("\n\n")[0]


class _ModeloFalso:
    """Chat model do LangChain: o structured output é ele mesmo, e só o ainvoke responde.

    `responder` é a corrotina que recebe o texto da linha e as mensagens.
    """

    def __init__(self, responder):
        self.responder = responder
        self.mensagens = []

    def with_structured_output(self, schema, *, include_raw=False):
        assert schema is Modelo
        assert include_raw
        return self

    async def ainvoke(self, mensagens):
        self.mensagens.append(mensagens)
        return await self.responder(_texto(mensagens), mensagens)

    def invoke(self, mensagens):
        msg = "o modo assíncrono não pode chamar invoke, que abre thread no LangChain"
        raise AssertionError(msg)


async def _eco(texto, _mensagens):
    return _resposta(texto)


def _roda(dados, responder=_eco, **kwargs):
    """Roda adataframeit num event loop novo, com o modelo falso."""
    modelo = _ModeloFalso(responder)
    kwargs.setdefault("track_tokens", False)
    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=modelo),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        resultado = asyncio.run(adataframeit(dados, Modelo, "{texto}", **kwargs))
    return resultado, modelo


def test_e_exportado_e_aceita_os_mesmos_parametros_de_dataframeit():
    assert "adataframeit" in pacote.__all__
    assert pacote.adataframeit is adataframeit
    assert inspect.iscoroutinefunction(adataframeit)
    sincrono = inspect.signature(dataframeit).parameters
    assincrono = inspect.signature(adataframeit).parameters
    assert list(assincrono) == list(sincrono)
    assert [p.default for p in assincrono.values()] == [p.default for p in sincrono.values()]
    assert [p.kind for p in assincrono.values()] == [p.kind for p in sincrono.values()]


@pytest.mark.parametrize("paralelo", [1, 3])
def test_devolve_o_mesmo_resultado_de_dataframeit(paralelo):
    df = pd.DataFrame({"texto": ["a", "b", "c", "d"], "outra": [1, 2, 3, 4]})

    def sincrono(text, *args, **kwargs):
        return {"data": {"campo1": text, "campo2": text.upper()}, "usage": None}

    with (
        patch("dataframeit.core.call_langchain", side_effect=sincrono),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        esperado = dataframeit(df, Modelo, "{texto}", track_tokens=False)

    resultado, _modelo = _roda(df, parallel_requests=paralelo)

    pd.testing.assert_frame_equal(resultado, esperado)


def test_roda_sem_threads_e_sem_time_sleep_com_linhas_em_voo_ao_mesmo_tempo(monkeypatch):
    """A garantia do Pyodide: Thread.start levanta, time.sleep bloquearia o loop.

    As linhas esperam o provider juntas, o que só acontece com concorrência real
    no event loop. A primeira tentativa de 'b' falha, e a espera da nova
    tentativa e o rate_limit_delay passam por asyncio.sleep.
    """
    em_voo = 0
    maximo_em_voo = 0
    tentativas = {}

    async def responder(texto, _mensagens):
        nonlocal em_voo, maximo_em_voo
        em_voo += 1
        maximo_em_voo = max(maximo_em_voo, em_voo)
        try:
            await asyncio.sleep(0.01)
            tentativas[texto] = tentativas.get(texto, 0) + 1
            if texto == "b" and tentativas[texto] == 1:
                msg = "instável"
                raise ProviderTransientError(msg)
            return _resposta(texto)
        finally:
            em_voo -= 1

    def sem_thread(self):
        msg = "can't start new thread"
        raise RuntimeError(msg)

    def sem_sleep(segundos):
        msg = f"time.sleep({segundos}) bloquearia o event loop"
        raise AssertionError(msg)

    monkeypatch.setattr(threading.Thread, "start", sem_thread)
    monkeypatch.setattr(time, "sleep", sem_sleep)
    entregues = []
    with pytest.warns(UserWarning, match="Tentativa 1/2 falhou"):
        resultado, _modelo = _roda(
            ["a", "b", "c", "d", "e", "f"],
            responder,
            parallel_requests=3,
            max_retries=2,
            base_delay=0.001,
            max_delay=0.001,
            rate_limit_delay=0.001,
            on_row_done=entregues.append,
        )

    assert maximo_em_voo == 3
    assert resultado["campo1"].tolist() == ["a", "b", "c", "d", "e", "f"]
    assert resultado["campo2"].tolist() == ["A", "B", "C", "D", "E", "F"]
    assert resultado["_error_details"].tolist() == [
        None,
        "Sucesso após 1 retry(s)",
        None,
        None,
        None,
        None,
    ]
    assert sorted(r.position for r in entregues) == [0, 1, 2, 3, 4, 5]
    assert all(r.success for r in entregues)


def test_sem_concorrencia_com_um_worker():
    em_voo = 0
    maximo_em_voo = 0

    async def responder(texto, _mensagens):
        nonlocal em_voo, maximo_em_voo
        em_voo += 1
        maximo_em_voo = max(maximo_em_voo, em_voo)
        await asyncio.sleep(0)
        em_voo -= 1
        return _resposta(texto)

    resultado, modelo = _roda(["a", "b", "c"], responder)

    assert maximo_em_voo == 1
    assert modelo.mensagens == ["a", "b", "c"]
    assert resultado["campo1"].tolist() == ["a", "b", "c"]


def test_pausas_usam_asyncio_sleep(monkeypatch):
    esperas = []

    async def espera(segundos):
        esperas.append(segundos)

    monkeypatch.setattr(core.asyncio, "sleep", espera)
    monkeypatch.setattr(errors.asyncio, "sleep", espera)
    chamadas = []

    async def responder(texto, _mensagens):
        chamadas.append(texto)
        if chamadas.count(texto) == 1 and texto == "a":
            msg = "instável"
            raise ProviderTransientError(msg)
        return _resposta(texto)

    with pytest.warns(UserWarning, match="Tentativa 1/3 falhou"):
        resultado, _modelo = _roda(
            ["a", "b"],
            responder,
            max_retries=3,
            base_delay=4.0,
            max_delay=4.0,
            rate_limit_delay=0.5,
        )

    assert resultado["_dataframeit_status"].tolist() == ["processed", "processed"]
    # O backoff de 4s com até 10% de jitter, e a pausa de cada linha bem-sucedida.
    assert len(esperas) == 3
    assert 4.0 <= esperas[0] <= 4.4
    assert esperas[1:] == [0.5, 0.5]


@pytest.mark.parametrize("paralelo", [1, 2])
def test_erro_que_esgota_as_tentativas_e_o_nao_recuperavel_viram_status_error(paralelo):
    async def responder(texto, _mensagens):
        if texto == "a":
            msg = "instável"
            raise ProviderTransientError(msg)
        if texto == "b":
            msg = "recusado"
            raise ProviderError(msg)
        return _resposta(texto)

    with (
        pytest.warns(UserWarning, match="Tentativa 1/2 falhou"),
        pytest.warns(UserWarning, match="Erro não-recuperável detectado"),
        pytest.warns(UserWarning, match="Falha ao processar linha 0"),
        pytest.warns(UserWarning, match="Falha ao processar linha 1"),
    ):
        resultado, _modelo = _roda(
            ["a", "b", "c"],
            responder,
            parallel_requests=paralelo,
            max_retries=2,
            base_delay=0,
            max_delay=0,
        )

    assert resultado["_dataframeit_status"].tolist() == ["error", "error", "processed"]
    detalhes = resultado["_error_details"].tolist()
    assert detalhes[0] == "[Falhou após 2 tentativa(s)] ProviderTransientError: instável"
    assert detalhes[1] == "[Erro não-recuperável] ProviderError: recusado"
    assert detalhes[2] is None


def test_aviso_da_nova_tentativa_aponta_a_chamada_do_usuario():
    chamadas = []

    async def responder(texto, _mensagens):
        chamadas.append(texto)
        if len(chamadas) == 1:
            msg = "instável"
            raise ProviderTransientError(msg)
        return _resposta(texto)

    modelo = _ModeloFalso(responder)

    async def principal():
        return await adataframeit(["a"], Modelo, "{texto}", max_retries=2, base_delay=0)

    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=modelo),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.warns(UserWarning, match="Tentativa 1/2 falhou") as avisos,
    ):
        asyncio.run(principal())

    (registro,) = [a for a in avisos if "Tentativa" in str(a.message)]
    assert registro.filename == __file__
    assert "await adataframeit([" in linecache.getline(registro.filename, registro.lineno)


def test_resposta_recusada_volta_ao_modelo_com_o_pedido_de_correcao():
    async def responder(texto, mensagens):
        if isinstance(mensagens, str):
            msg = "sem JSON"
            raise OutputParserException(msg, llm_output="não é JSON")
        return _resposta(texto)

    with pytest.warns(UserWarning, match="Tentativa 1/2 falhou"):
        resultado, modelo = _roda(["a"], responder, max_retries=2, base_delay=0, max_delay=0)

    assert resultado["campo1"].tolist() == ["a"]
    primeira, segunda = modelo.mensagens
    assert primeira == "a"
    assert segunda[0] == ("human", "a")
    assert segunda[1] == ("ai", "não é JSON")
    assert "não passou na validação" in segunda[2][1]


def test_uso_das_tentativas_recusadas_entra_no_resumo_da_linha_que_falha(caplog):
    uso = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}

    async def responder(texto, _mensagens):
        resposta = _resposta(texto, uso)
        resposta["parsed"] = None
        return resposta

    with (
        caplog.at_level(logging.INFO, logger="dataframeit.stats"),
        pytest.warns(UserWarning, match="Tentativa 1/2 falhou"),
        pytest.warns(UserWarning, match="Falha ao processar linha 0"),
    ):
        resultado, _modelo = _roda(
            ["a"], responder, max_retries=2, base_delay=0, max_delay=0, track_tokens=True
        )

    assert resultado["_dataframeit_status"].tolist() == ["error"]
    assert "Total de tokens: 20" in caplog.text


@pytest.mark.parametrize(("paralelo", "vazao"), [(1, False), (2, True)])
def test_resumo_de_uso_segue_o_modo_sequencial_ou_o_paralelo(caplog, paralelo, vazao):
    uso = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}

    async def responder(texto, _mensagens):
        await asyncio.sleep(0.001)
        return _resposta(texto, uso)

    with caplog.at_level(logging.INFO, logger="dataframeit.stats"):
        resultado, _modelo = _roda(
            ["a", "b"], responder, parallel_requests=paralelo, track_tokens=True
        )

    assert resultado["_input_tokens"].tolist() == [7, 7]
    assert "Total de tokens: 20" in caplog.text
    assert ("METRICAS DE THROUGHPUT" in caplog.text) is vazao
    assert ("Requisicoes: 2" in caplog.text) is vazao


@pytest.mark.parametrize("paralelo", [1, 2])
def test_on_row_done_recebe_cada_linha_e_aceita_corrotina(paralelo):
    entregues = []

    async def ao_terminar(resultado):
        await asyncio.sleep(0)
        entregues.append(resultado)

    async def responder(texto, _mensagens):
        if texto == "b":
            msg = "recusado"
            raise ProviderError(msg)
        return _resposta(texto)

    with (
        pytest.warns(UserWarning, match="Erro não-recuperável detectado"),
        pytest.warns(UserWarning, match="Falha ao processar linha y"),
        pytest.warns(UserWarning, match="sem texto não vão ao LLM"),
    ):
        _roda(
            {"x": "a", "y": "b", "z": None},
            responder,
            parallel_requests=paralelo,
            on_row_done=ao_terminar,
        )

    por_posicao = sorted(entregues, key=lambda r: r.position)
    assert all(isinstance(r, RowResult) for r in por_posicao)
    assert [r.index for r in por_posicao] == ["x", "y", "z"]
    assert [r.position for r in por_posicao] == [0, 1, 2]
    assert [r.success for r in por_posicao] == [True, False, False]
    assert por_posicao[0].values == {"campo1": "a", "campo2": "A"}
    assert por_posicao[0].error is None
    assert por_posicao[1].values == {}
    assert por_posicao[1].error == "[Erro não-recuperável] ProviderError: recusado"
    assert por_posicao[2].error == "Texto ausente"
    assert all(r.elapsed_seconds >= 0 for r in por_posicao)


def test_falha_do_on_row_done_vira_aviso_e_a_execucao_segue():
    def ao_terminar(resultado):
        msg = "callback quebrado"
        raise RuntimeError(msg)

    with pytest.warns(UserWarning, match="on_row_done falhou na linha 0: RuntimeError"):
        resultado, _modelo = _roda(["a"], on_row_done=ao_terminar)

    assert resultado["campo1"].tolist() == ["a"]


@pytest.mark.parametrize("paralelo", [1, 2])
def test_abort_interrompe_e_deixa_as_linhas_nao_alcancadas_sem_status(paralelo):
    async def responder(texto, _mensagens):
        if texto == "b":
            msg = "cota esgotada"
            raise ProviderAbortError(msg)
        if texto == "a":
            # Com dois workers, 'a' termina depois de 'b' interromper.
            await asyncio.sleep(0.01)
        return _resposta(texto)

    with pytest.warns(UserWarning, match="Execução interrompida: ProviderAbortError: cota"):
        resultado, modelo = _roda(
            pd.DataFrame({"texto": ["a", "b", "c", "d"]}), responder, parallel_requests=paralelo
        )

    assert resultado["_dataframeit_status"].tolist() == ["processed", None, None, None]
    assert resultado["campo1"].tolist() == ["a", None, None, None]
    assert modelo.mensagens == ["a", "b"]


def test_rate_limit_reduz_os_workers_e_pausa_as_linhas_seguintes(monkeypatch):
    """'a' bate no rate limit com 'b' em voo: o teto cai para 1, e 'c' e 'd' pausam antes.

    O worker de 'b' está acima do novo teto e sai depois de terminá-la; o de
    'a' segue sozinho, e cada linha que ele começa na janela do rate limit
    espera a pausa.
    """
    dorme = asyncio.sleep
    esperas = []

    async def espera(segundos):
        esperas.append(segundos)
        await dorme(0)

    monkeypatch.setattr(core.asyncio, "sleep", espera)
    eventos = []
    reduziu = asyncio.Event()

    async def responder(texto, _mensagens):
        eventos.append(f"{texto} início")
        if texto == "a":
            await dorme(0)
            reduziu.set()
            eventos.append("a fim")
            msg = "429 too many requests"
            raise ProviderOverloadedError(msg)
        if texto == "b":
            await reduziu.wait()
        eventos.append(f"{texto} fim")
        return _resposta(texto)

    async def principal():
        return await adataframeit(
            ["a", "b", "c", "d"],
            Modelo,
            "{texto}",
            parallel_requests=2,
            max_retries=1,
            track_tokens=False,
        )

    modelo = _ModeloFalso(responder)
    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=modelo),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.warns(UserWarning, match="Falha ao processar linha 0"),
        pytest.warns(UserWarning, match="Rate limit detectado! Reduzindo workers de 2 para 1"),
        pytest.warns(UserWarning, match="Workers reduzidos por rate limit: de 2 para 1") as avisos,
    ):
        resultado = asyncio.run(principal())

    assert len([a for a in avisos if "Rate limit detectado" in str(a.message)]) == 1
    assert resultado["_dataframeit_status"].tolist() == [
        "error",
        "processed",
        "processed",
        "processed",
    ]
    assert esperas == [core._RATE_LIMIT_PAUSE_SECONDS] * 2
    # 'a' e 'b' estavam em voo juntas; 'c' só começa com o teto já em 1.
    assert eventos.index("b início") < eventos.index("a fim")
    assert eventos.index("c início") > eventos.index("a fim")


def test_interrupcao_durante_a_pausa_do_rate_limit_deixa_a_linha_sem_chamar(monkeypatch):
    """'a' bate no rate limit com 'b' em voo; 'c' pausa, e 'b' interrompe durante a pausa.

    'c' já saiu da fila, mas não chega ao provider: fica sem status, como as
    linhas que a interrupção não alcançou.
    """
    dorme = asyncio.sleep
    abortou = asyncio.Event()
    reduziu = asyncio.Event()

    async def espera(_segundos):
        await abortou.wait()

    monkeypatch.setattr(core.asyncio, "sleep", espera)

    async def responder(texto, _mensagens):
        if texto == "a":
            await dorme(0)
            reduziu.set()
            msg = "429 too many requests"
            raise ProviderOverloadedError(msg)
        await reduziu.wait()
        abortou.set()
        msg = "cota esgotada"
        raise ProviderAbortError(msg)

    async def principal():
        return await adataframeit(
            ["a", "b", "c"],
            Modelo,
            "{texto}",
            parallel_requests=2,
            max_retries=1,
            track_tokens=False,
        )

    modelo = _ModeloFalso(responder)
    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=modelo),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.warns(UserWarning, match="Falha ao processar linha 0"),
        pytest.warns(UserWarning, match="Rate limit detectado! Reduzindo workers de 2 para 1"),
        pytest.warns(UserWarning, match="Execução interrompida"),
        pytest.warns(UserWarning, match="Workers reduzidos"),
    ):
        resultado = asyncio.run(principal())

    assert modelo.mensagens == ["a", "b"]
    assert resultado["_dataframeit_status"].tolist() == ["error", None, None]


def test_falha_fora_do_tratamento_da_linha_marca_a_linha():
    def gravacao_que_falha(*args, **kwargs):
        msg = "gravação"
        raise RuntimeError(msg)

    with (
        patch("dataframeit.core._record_missing_text", side_effect=gravacao_que_falha),
        pytest.warns(UserWarning, match="sem texto não vão ao LLM"),
        pytest.warns(UserWarning, match="Falha ao processar linha 1"),
    ):
        resultado, _modelo = _roda(pd.DataFrame({"texto": ["a", None, "c"]}), parallel_requests=2)

    assert resultado["_dataframeit_status"].tolist() == ["processed", "error", "processed"]
    assert resultado["_error_details"].iloc[1] == (
        "[Falha ao registrar o erro] RuntimeError: gravação"
    )


@pytest.mark.parametrize("paralelo", [1, 2])
def test_cancelamento_grava_o_checkpoint_e_a_retomada_faz_so_o_que_faltou(tmp_path, paralelo):
    checkpoint = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c", "d", "e"]})
    travadas = []
    nunca = asyncio.Event()
    todas_travadas = asyncio.Event()

    async def trava_em_c(texto, _mensagens):
        if texto >= "c":
            travadas.append(texto)
            if len(travadas) == paralelo:
                todas_travadas.set()
            await nunca.wait()
        return _resposta(texto)

    async def principal():
        tarefa = asyncio.create_task(
            adataframeit(
                df,
                Modelo,
                "{texto}",
                parallel_requests=paralelo,
                batch_size=10,
                checkpoint_path=checkpoint,
                track_tokens=False,
            )
        )
        await todas_travadas.wait()
        tarefa.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarefa

    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=_ModeloFalso(trava_em_c)),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        asyncio.run(principal())

    gravado = pd.read_csv(checkpoint)
    assert gravado["_dataframeit_status"].tolist()[:2] == ["processed", "processed"]
    assert gravado["_dataframeit_status"].iloc[2:].isna().all()

    retomado, modelo = _roda(df, batch_size=10, checkpoint_path=checkpoint)

    assert modelo.mensagens == ["c", "d", "e"]
    assert retomado["campo1"].tolist() == ["a", "b", "c", "d", "e"]


def test_checkpoint_e_gravado_a_cada_lote(tmp_path):
    checkpoint = tmp_path / "ckpt.parquet"
    pytest.importorskip("pyarrow")
    gravacoes = []
    original = core._try_save_checkpoint

    def conta(df, ckpt):
        gravacoes.append(int(df["_dataframeit_status"].notna().sum()))
        return original(df, ckpt)

    with patch("dataframeit.core._try_save_checkpoint", side_effect=conta):
        resultado, _modelo = _roda(
            ["a", "b", "c", "d", "e"],
            parallel_requests=2,
            batch_size=2,
            checkpoint_path=checkpoint,
        )

    assert resultado["campo1"].tolist() == ["a", "b", "c", "d", "e"]
    # Dois lotes de 2 linhas e a cauda de 1.
    assert gravacoes == [2, 4, 5]
    assert pd.read_parquet(checkpoint)["campo1"].tolist() == ["a", "b", "c", "d", "e"]


def test_entrada_vazia_devolve_sem_chamar_o_modelo():
    resultado, modelo = _roda([])

    assert resultado.empty
    assert modelo.mensagens == []


@pytest.mark.parametrize(
    ("kwargs", "mensagem"),
    [
        ({"provider": "codex"}, "adataframeit não suporta provider='codex'"),
        ({"provider": "claude_code"}, "adataframeit não suporta provider='claude_code'"),
        ({"use_search": True}, r"adataframeit não suporta busca web \(use_search=True\)"),
    ],
)
def test_backend_sem_caminho_assincrono_levanta_antes_de_validar(kwargs, mensagem):
    # Sem `questions`, dataframeit levantaria por ela; a recusa vem antes.
    with pytest.raises(ValueError, match=mensagem):
        asyncio.run(adataframeit(["a"], **kwargs))
