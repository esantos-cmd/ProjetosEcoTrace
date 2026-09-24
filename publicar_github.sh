#!/usr/bin/env bash
# Publica a pasta Segmentacao no repositório ProjetosEcoTrace do GitHub.
set -e
cd "$(dirname "$0")"

REPO_URL="https://github.com/esantos-cmd/ProjetosEcoTrace.git"

# Nome/e-mail do Git (só configura se ainda não existir)
git config --global user.name  >/dev/null || git config --global user.name  "Eddy Santos"
git config --global user.email >/dev/null || git config --global user.email "e.santos@ecotrace.info"

[ -d .git ] || git init
git add .
git commit -m "Primeiro commit: projeto de segmentação de carnes" || echo "Nada novo para commitar."
git branch -M main

if git remote | grep -q '^origin$'; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

echo
echo "Arquivos que serão enviados:"
git ls-files
echo
git push -u origin main
echo
echo "Pronto! Veja em: https://github.com/esantos-cmd/ProjetosEcoTrace"
