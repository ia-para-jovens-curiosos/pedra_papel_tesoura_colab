# 🎮 IA para Jovens Curiosos — Pedra, Papel e Tesoura (Google Colab)

[![GitHub Repo](https://img.shields.io/badge/GitHub-ia--para--jovens--curiosos%2Fpedra__papel__tesoura__colab-blue?logo=github)](https://github.com/ia-para-jovens-curiosos/pedra_papel_tesoura_colab)
[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ia-para-jovens-curiosos/pedra_papel_tesoura_colab/blob/master/sample.ipynb)

Um projeto para crianças treinarem sua própria Inteligência Artificial, usando a câmera do computador, para jogar pedra, papel e tesoura contra o computador — direto do navegador, sem instalar nada.

Esta é a versão **Google Colab** do projeto [`pedra_papel_tesoura_jupyter`](https://github.com/ia-para-jovens-curiosos/pedra_papel_tesoura_jupyter), que usa a webcam local. Use esta versão se não quiser instalar Python/TensorFlow na sua máquina.

## Como abrir

Clique no botão **"Abrir no Colab"** acima (ou abra `sample.ipynb` diretamente em [colab.research.google.com](https://colab.research.google.com)) e execute as células de cima para baixo, com `Shift + Enter`.

Quando o navegador pedir permissão para usar a câmera, clique em **Permitir**.

O notebook guia você por quatro passos:

1. **Tirar fotos** — tirar fotos da sua mão fazendo pedra, papel e tesoura
2. **Treinar** — ensinar o robô a reconhecer cada jogada
3. **Testar** — conferir se o robô aprendeu direito
4. **Jogar** — jogar pedra, papel e tesoura contra o computador

## Como funciona por baixo dos panos

Todo o TensorFlow e o OpenCV ficam escondidos dentro do arquivo `ia_pedra_papel_tesoura.py`, que oferece só quatro funções simples em português:

- `tirar_fotos(jogada, quantidade)`
- `treinar()`
- `testar(caminho_da_imagem=None)`
- `jogar()`

As fotos tiradas ficam salvas em `data/pedra`, `data/papel` e `data/tesoura`, e o robô treinado é salvo em `models/`.

Como o Colab roda numa máquina na nuvem (sem acesso direto à webcam do seu computador), a captura de fotos usa a câmera do navegador através de JavaScript: em vez de pressionar ESPAÇO/ESC numa janela do OpenCV, você clica nos botões **"📸 Tirar foto"** e **"⏹ Parar"** que aparecem no próprio notebook.

## ⚠️ Atenção: o Colab não guarda seus arquivos

As máquinas do Colab são temporárias — quando a sessão termina (ou fica muito tempo parada), as fotos em `data/` e o robô treinado em `models/` são apagados. Se quiser guardar seu progresso entre sessões, baixe as pastas `data/` e `models/` pelo painel de arquivos do Colab (ícone de pasta 📁 à esquerda) antes de encerrar.

## Ambiente

Este notebook já roda com o ambiente padrão do Google Colab — não é preciso instalar nada. O arquivo `requirements.txt` serve apenas de referência, caso você queira reproduzir o mesmo ambiente localmente (sem usar a câmera, já que a captura via JavaScript só funciona no Colab).

É necessário ter uma câmera conectada e permitir o acesso a ela pelo navegador.
