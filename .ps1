# setup_env.ps1
Write-Host "🚀 正在创建虚拟环境 venv..."
python -m venv venv

Write-Host "🔧 正在激活虚拟环境..."
. .\venv\Scripts\Activate.ps1

Write-Host "📦 正在升级 pip..."
python -m pip install --upgrade pip

Write-Host "📥 正在安装依赖包..."
pip install opencv-python mediapipe==0.10.0 pandas matplotlib

Write-Host "✅ 环境准备完成，可以开始开发啦！"