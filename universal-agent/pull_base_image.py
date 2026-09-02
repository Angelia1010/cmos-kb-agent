# -*- coding: utf-8 -*-
"""免 Docker 拉取 OCI/Docker 镜像并打包为 `docker load` 兼容 tar。

用途:在无 Docker 引擎的机器上,从可达的 Registry(如 DaoCloud 镜像源)
     拉取指定平台(如 linux/amd64)的镜像,产出可直接在生产环境
     `docker load -i xxx.tar` 导入的归档文件。

流程:
  1) 匿名 token(www-authenticate 指示的 realm)
  2) 拉 tag 的 manifest index;按 os/architecture 选出目标平台 manifest
  3) 拉单架构 manifest,得 config + layers 摘要
  4) 逐层下载(压缩),流式解压为 layer.tar 并校验 sha256 == diff_id
  5) 组装 docker-save 格式:manifest.json / <config>.json / <diffid>/layer.tar / repositories

仅用标准库,无第三方依赖。
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request

REGISTRY = "https://docker.m.daocloud.io"
REPO = "library/python"
TAG = "3.12-slim"
WANT_OS = "linux"
WANT_ARCH = "amd64"
OUT_TAR = os.path.join(os.path.dirname(__file__), "..", "dist",
                       f"python-{TAG}-{WANT_ARCH}.tar")

ACCEPT_INDEX = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
])
ACCEPT_MANIFEST = ", ".join([
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


class _Redirect(Exception):
    def __init__(self, location: str, status: int):
        self.location = location
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁用自动重定向,改为抛 _Redirect,由调用方手动跟随(便于剥掉鉴权头)。"""
    def http_error_302(self, req, fp, code, msg, headers):
        raise _Redirect(headers.get("Location"), code)
    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


_opener = urllib.request.build_opener(_NoRedirect())


def http_open(url: str, headers: dict | None = None, timeout: int = 300,
              max_redirects: int = 6):
    """GET url。手动跟随重定向;跳往后端存储后不再携带 Authorization。"""
    cur_headers = dict(headers or {})
    # Cloudflare 会拦截默认 Python-urllib UA(403),统一换成常见客户端 UA
    cur_headers.setdefault("User-Agent", "curl/8.5.0")
    cur_url = url
    for _ in range(max_redirects + 1):
        req = urllib.request.Request(cur_url, headers=cur_headers)
        try:
            return _opener.open(req, timeout=timeout)
        except _Redirect as rd:
            if not rd.location:
                raise SystemExit(f"重定向无 Location: {cur_url}")
            nxt = urllib.parse.urljoin(cur_url, rd.location)
            # 跳出 registry 主机后剥掉鉴权头(后端多为预签名 URL)
            if urllib.parse.urlparse(nxt).netloc != urllib.parse.urlparse(cur_url).netloc:
                cur_headers.pop("Authorization", None)
            cur_url = nxt
    raise SystemExit(f"重定向次数超限: {url}")


def get_token() -> str:
    url = ("https://m.daocloud.io/auth/token"
           f"?service=docker.m.daocloud.io&scope=repository:{REPO}:pull")
    with http_open(url) as r:
        return json.loads(r.read())["token"]


def auth_hdr(token: str, accept: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": accept}


def fetch_manifest(token: str, ref: str, accept: str):
    url = f"{REGISTRY}/v2/{REPO}/manifests/{ref}"
    with http_open(url, auth_hdr(token, accept)) as r:
        return json.loads(r.read()), r.headers.get("Content-Type", "")


def select_platform(index: dict) -> str:
    for m in index.get("manifests", []):
        p = m.get("platform", {})
        if p.get("os") == WANT_OS and p.get("architecture") == WANT_ARCH:
            variant = p.get("variant")
            if WANT_ARCH != "arm" or not variant:  # amd64 无 variant 要求
                print(f"  命中平台: os={p.get('os')} arch={p.get('architecture')}"
                      f"{' variant=' + variant if variant else ''}")
                return m["digest"]
    raise SystemExit("索引中未找到目标平台 %s/%s" % (WANT_OS, WANT_ARCH))


def fetch_blob(token: str, digest: str, dest: str | None = None):
    """下载 blob;dest 为 None 时返回 bytes,否则流式写文件并返回 sha256。"""
    url = f"{REGISTRY}/v2/{REPO}/blobs/{digest}"
    h = hashlib.sha256()
    with http_open(url, {"Authorization": f"Bearer {token}"}) as r:
        if dest is None:
            data = r.read()
            return data
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                f.write(chunk)
    return h.hexdigest()


def gunzip_to(src: str, dst: str) -> str:
    """解压 gzip 到 dst,返回解压内容的 sha256(用于比对 diff_id)。"""
    h = hashlib.sha256()
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            fout.write(chunk)
    return h.hexdigest()


def main():
    work = os.path.join(os.path.dirname(OUT_TAR), "_pull_tmp")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TAR), exist_ok=True)

    print(f"[1/5] 获取匿名 token …")
    token = get_token()
    print(f"      token ok ({len(token)} chars)")

    print(f"[2/5] 拉取 {REPO}:{TAG} 的多架构索引 …")
    index, ct = fetch_manifest(token, TAG, ACCEPT_INDEX)
    if "manifests" in index:  # index / manifest list
        target_digest = select_platform(index)
        print(f"      单架构 manifest digest: {target_digest}")
        manifest, mct = fetch_manifest(token, target_digest, ACCEPT_MANIFEST)
    else:  # 直接就是单架构 manifest
        manifest, mct = index, ct
    print(f"      manifest mediaType: {mct}")

    cfg_desc = manifest["config"]
    layers = manifest["layers"]
    print(f"[3/5] 下载 config blob({cfg_desc['size']} B)与 {len(layers)} 个层 …")
    cfg_bytes = fetch_blob(token, cfg_desc["digest"])
    config = json.loads(cfg_bytes)
    diff_ids = config["rootfs"]["diff_ids"]  # ["sha256:...", ...] 未压缩摘要
    if len(diff_ids) != len(layers):
        raise SystemExit("层数量与 diff_ids 不一致,镜像异常")
    cfg_hex = cfg_desc["digest"].split(":", 1)[1]

    print("[4/5] 逐层下载并解压校验 …")
    layer_paths = []
    for i, (ldesc, diff_id) in enumerate(zip(layers, diff_ids), 1):
        want_hex = diff_id.split(":", 1)[1]
        gz_path = os.path.join(work, f"layer{i}.tar.gz")
        got = fetch_blob(token, ldesc["digest"], gz_path)
        if got != ldesc["digest"].split(":", 1)[1]:
            raise SystemExit(f"层 {i} 压缩摘要不符: {got}")
        out_dir = os.path.join(work, want_hex)
        os.makedirs(out_dir, exist_ok=True)
        layer_tar = os.path.join(out_dir, "layer.tar")
        unc_hex = gunzip_to(gz_path, layer_tar)
        if unc_hex != want_hex:
            raise SystemExit(f"层 {i} 解压后 sha256 与 diff_id 不符: {unc_hex} != {want_hex}")
        with open(os.path.join(out_dir, "VERSION"), "w") as f:
            f.write("1.0")
        layer_paths.append(f"{want_hex}/layer.tar")
        size_mb = os.path.getsize(layer_tar) / 1048576
        print(f"      层 {i}/{len(layers)} ok  diff_id={want_hex[:12]}  {size_mb:.1f} MB")
        os.remove(gz_path)

    print("[5/5] 组装 docker-save tar …")
    manifest_json = [{
        "Config": f"{cfg_hex}.json",
        "RepoTags": [f"python:{TAG}"],
        "Layers": layer_paths,
    }]
    repositories = {"python": {TAG: layer_paths[-1].split("/")[0]}}

    with tarfile.open(OUT_TAR, "w") as tf:
        def add_bytes(name: str, data: bytes):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        add_bytes("manifest.json", json.dumps(manifest_json).encode())
        add_bytes("repositories", json.dumps(repositories).encode())
        add_bytes(f"{cfg_hex}.json", cfg_bytes)
        for lp in layer_paths:
            d = lp.split("/")[0]
            tf.add(os.path.join(work, d, "layer.tar"), arcname=lp)
            tf.add(os.path.join(work, d, "VERSION"), arcname=f"{d}/VERSION")

    shutil.rmtree(work)
    out_mb = os.path.getsize(OUT_TAR) / 1048576
    print(f"\n完成: {os.path.abspath(OUT_TAR)}  ({out_mb:.1f} MB)")
    print(f"  镜像: python:{TAG}  平台: {WANT_OS}/{WANT_ARCH}  层数: {len(layers)}")


if __name__ == "__main__":
    main()
