# GitHub Secrets 配置指南

为了启用完整的CI/CD功能，需要在GitHub仓库中配置以下Secrets：

## 必需的Secrets

### 1. PyPI发布
- **PYPI_API_TOKEN**: PyPI API令牌
  - 获取方式：登录 [PyPI](https://pypi.org)，进入Account Settings > API tokens
  - 创建新的API令牌，复制token值
  - 在GitHub仓库设置中添加：`Settings > Secrets and variables > Actions > New repository secret`

### 2. TestPyPI发布
- **TEST_PYPI_API_TOKEN**: TestPyPI API令牌
  - 获取方式：登录 [TestPyPI](https://test.pypi.org)，进入Account Settings > API tokens
  - 创建新的API令牌，复制token值

## 可选的Secrets

### 3. Docker Hub发布
- **DOCKER_USERNAME**: Docker Hub用户名
- **DOCKER_PASSWORD**: Docker Hub密码或访问令牌
  - 获取方式：登录 [Docker Hub](https://hub.docker.com)，进入Account Settings > Security
  - 创建新的Access Token，复制token值

### 4. Codecov代码覆盖率
- **CODECOV_TOKEN**: Codecov令牌
  - 获取方式：登录 [Codecov](https://codecov.io)，连接GitHub仓库
  - 在仓库设置中获取token值

## 配置步骤

1. 进入GitHub仓库页面
2. 点击 `Settings` 标签
3. 在左侧菜单中点击 `Secrets and variables` > `Actions`
4. 点击 `New repository secret`
5. 输入Secret名称和值
6. 点击 `Add secret`

## 验证配置

配置完成后，推送代码或创建标签时，GitHub Actions会自动运行：

- **推送到main分支**：运行CI测试
- **创建v*标签**：运行完整发布流程

## 故障排除

### CI失败
- 检查Python版本配置
- 确保所有依赖都正确安装

### Docker发布失败
- 检查DOCKER_USERNAME和DOCKER_PASSWORD是否正确
- 确保Docker Hub账户有推送权限

### PyPI发布失败
- 检查PYPI_API_TOKEN是否有效
- 确保包名在PyPI上可用

### Codecov上传失败
- 检查CODECOV_TOKEN是否正确
- 确保coverage.xml文件生成成功
