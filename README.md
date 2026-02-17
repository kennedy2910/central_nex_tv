# Central-Nex v9

- Admin web (curadoria/cadastro)
- API para Edge puxar lista via **X-API-KEY**
- Playlist IPTV (M3U) por Edge, agrupada por Provider
- Timeline linear calculada na CENTRAL para canais `youtube_linear`

## Fluxo principal (novo)
1. Cadastrar canais livremente
2. Montar uma **Grade de Distribuicao** selecionando canais
3. Vincular cada **Edge** a uma grade especifica
4. Sem grade vinculada, o edge usa automaticamente `grade_geral_auto` (todos os canais ativos)
5. Conteudo (URLs) de canal `youtube_linear` e gerenciado ao abrir o canal no admin

## Filtros no Admin
- Canais: filtro por **Categoria** e **Grade**
- Edges: filtro por **Pais -> Estado -> Cidade**

## Fluxo avancado (opcional)
- Para canais `youtube_linear`, use playlists por canal + `edge_programming`
- A CENTRAL devolve `now` e `offset`

## Instalar
```bash
chmod +x scripts/*.sh
./scripts/install.sh
```

Admin: http://SEU_IP:9000/

## Zerar tudo
```bash
./scripts/uninstall.sh
```

## API (para o Edge)
```bash
curl -H "X-API-KEY: edge_xxx" http://SEU_IP:9000/api/edge/channels
```

## Playlist IPTV
```bash
curl "http://SEU_IP:9000/iptv/edge.m3u?api_key=edge_xxx"
```

> Caminho HLS esperado no Edge:
`{hls_base_url}/{provider_id}/{channel_id}/index.m3u8`
