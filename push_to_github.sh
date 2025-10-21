#!/bin/bash
# LeapHand项目GitHub推送脚本
# 用于解决网络连接问题

echo "=== LeapHand项目GitHub推送助手 ==="
echo "仓库: https://github.com/Sokinskes/leaphand-gen3.git"
echo ""

# 检查当前目录
if [ ! -d ".git" ]; then
    echo "❌ 错误: 这不是一个Git仓库"
    exit 1
fi

# 检查远程仓库
echo "📡 检查远程仓库配置..."
git remote -v

# 禁用代理和SSL验证
echo ""
echo "🔧 配置Git网络设置..."
git config --global --unset-all http.proxy 2>/dev/null
git config --global --unset-all https.proxy 2>/dev/null
git config --global http.sslVerify false
git config --global http.postBuffer 524288000

echo "✅ Git配置已更新"

# 测试连接
echo ""
echo "🌐 测试GitHub连接..."
if timeout 10 git ls-remote --heads https://github.com/Sokinskes/leaphand-gen3.git >/dev/null 2>&1; then
    echo "✅ 连接成功，准备推送..."
    git push -u origin main
    echo "🎉 推送完成！"
else
    echo "❌ 连接仍然失败"
    echo ""
    echo "💡 建议解决方案:"
    echo "1. 检查网络连接"
    echo "2. 尝试使用VPN"
    echo "3. 使用备用压缩包手动上传"
    echo "4. 联系网络管理员"
fi