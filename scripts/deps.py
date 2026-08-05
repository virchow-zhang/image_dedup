"""解析 dedup_core.exe 的 DLL 依赖闭包, 输出缺失依赖清单。

用法: python scripts/deps.py
"""
import os
import re
import subprocess
import sys

DUMPBIN = r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\dumpbin.exe'
LIBS = r'C:\Users\zhang\image_dedup\libs'
EXE = r'C:\Users\zhang\image_dedup\dedup_core.exe'

SYSTEM = {
    'KERNEL32.dll', 'USER32.dll', 'ADVAPI32.dll', 'SHELL32.dll', 'GDI32.dll',
    'WS2_32.dll', 'MSVCP140.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll',
    'ucrtbase.dll', 'bcrypt.dll', 'ole32.dll', 'comdlg32.dll', 'OLEAUT32.dll',
    'SHLWAPI.dll', 'WINMM.dll', 'MSIMG32.dll', 'COMDLG32.dll', 'WTSAPI32.dll',
    'IPHLPAPI.dll', 'UxTheme.dll', 'dwmapi.dll', 'SECUR32.dll', 'WININET.dll',
    'POWRPROF.dll', 'WSOCK32.dll', 'NTDLL.dll', 'PSAPI.dll', 'SHCORE.dll',
    'DBGHELP.dll', 'WINSPOOL.DRV', 'VERSION.dll', 'MSVCP140_1.dll',
    'MSVCP140_2.dll', 'CONCRT140.dll', 'VCCORLIB140.dll', 'mpr.dll',
    'NETAPI32.dll', 'secur32.dll', 'CFGMGR32.dll', 'setupapi.dll',
    'crypt32.dll', 'wldap32.dll', 'odbc32.dll', 'netapi32.dll',
    'DSUTILS.dll', 'd3d11.dll', 'dxgi.dll', 'dwrite.dll', 'd2d1.dll',
}


def deps_of(dll: str):
    r = subprocess.run([DUMPBIN, '/dependents', dll], capture_output=True,
                       text=True, errors='replace')
    out = []
    for m in re.finditer(r'^\s+([\w.\-]+\.dll)', r.stdout, re.M):
        out.append(m.group(1))
    return out


def resolve(start: str):
    queue = [start]
    seen = set()
    missing = {}
    found = []
    while queue:
        dll = queue.pop(0)
        name = os.path.basename(dll)
        if name in seen:
            continue
        seen.add(name)
        for dep in deps_of(dll):
            if dep.upper() in {s.upper() for s in SYSTEM} or dep.startswith('api-ms-win-'):
                continue
            loc = os.path.join(LIBS, dep)
            if os.path.exists(loc):
                if dep not in seen:
                    queue.append(loc)
                    found.append(dep)
            else:
                missing.setdefault(dep, name)
    return found, missing


if __name__ == '__main__':
    import shutil
    CONDABIN = r'C:\Users\zhang\anaconda3\envs\dedup_build\Library\bin'
    for _ in range(20):  # 迭代补齐
        found, missing = resolve(EXE)
        if not missing:
            break
        for dep in list(missing):
            src = os.path.join(CONDABIN, dep)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(LIBS, dep))
                print(f'补齐: {dep}')
    found, missing = resolve(EXE)
    print(f'需要 {len(found)} 个非系统 DLL')
    print('缺失:')
    for d, parent in sorted(missing.items()):
        print(f'  {d}  (由 {parent} 引用)')
    if not missing:
        print('依赖完整 ✓')
