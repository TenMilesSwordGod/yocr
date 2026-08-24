# yocr —— 开发 / Docker / systemd 部署任务
# 用法: make help

# ----------------------------- 可调参数 ---------------------------------
HOST            ?= 0.0.0.0
PORT            ?= 8000
DEVICE          ?= cpu                 # cpu | cuda:0
OCR_LANG        ?= ch
COMPOSE         ?= docker compose
SYSTEMD_UNIT    ?= /etc/systemd/system/yocr.service
ENV_FILE        ?= /etc/yocr/yocr.env  # 可选的额外环境变量文件(不存在则忽略)
RUN_USER        ?= $(shell echo $${SUDO_USER:-$$USER})
RUN_GROUP       ?= $(shell id -gn $${SUDO_USER:-$$USER} 2>/dev/null || echo $(RUN_USER))

.DEFAULT_GOAL := help
.PHONY: help install run test clean health \
        docker-build docker-up docker-down docker-restart docker-logs \
        systemd-install systemd-uninstall systemd-start systemd-stop \
        systemd-restart systemd-status systemd-logs

help: ## 显示本帮助
	@echo "yocr make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "可调参数(示例): make run PORT=9000 DEVICE=cuda:0"

# ------------------------------ 本地开发 --------------------------------
install: ## 安装依赖 (uv sync)
	uv sync --frozen

run: ## 前台启动服务 (HOST/PORT 可覆盖)
	uv run yocr serve --host $(HOST) --port $(PORT)

test: ## 运行单元测试
	uv run pytest tests/ -q

health: ## 探活: curl /api/v1/healthz
	curl -s --noproxy '*' http://127.0.0.1:$(PORT)/api/v1/healthz && echo

clean: ## 清理测试缓存
	rm -rf .pytest_cache tests/__pycache__ src/yocr/__pycache__

# ------------------------------- Docker ---------------------------------
docker-build: ## 构建镜像
	$(COMPOSE) build

docker-up: ## 构建并后台启动容器 (含重建)
	$(COMPOSE) up -d --build

docker-down: ## 停止并移除容器
	$(COMPOSE) down

docker-restart: ## 重启容器
	$(COMPOSE) restart

docker-logs: ## 跟踪容器日志
	$(COMPOSE) logs -f

# ------------------------------- systemd --------------------------------
# 以"当前仓库目录 + .venv"方式常驻运行；sudo 安装，服务以当前用户身份运行。
systemd-install: install ## 渲染 unit 并安装(需 sudo)，完成后需手动 start
	@sed -e "s|__APP_DIR__|$(CURDIR)|g" \
	     -e "s|__RUN_USER__|$(RUN_USER)|g" \
	     -e "s|__RUN_GROUP__|$(RUN_GROUP)|g" \
	     -e "s|__HOST__|$(HOST)|g" \
	     -e "s|__PORT__|$(PORT)|g" \
	     -e "s|__DEVICE__|$(DEVICE)|g" \
	     -e "s|__OCR_LANG__|$(OCR_LANG)|g" deploy/yocr.service | sudo tee $(SYSTEMD_UNIT) > /dev/null
	sudo systemctl daemon-reload
	@echo "已安装 $(SYSTEMD_UNIT)  ->  make systemd-start 启动"

systemd-start: ## 启动并设置开机自启
	sudo systemctl enable --now yocr
	@sleep 2; systemctl --no-pager status yocr | head -8 || true

systemd-stop: ## 停止服务并取消开机自启
	sudo systemctl disable --now yocr

systemd-restart: ## 重启服务
	sudo systemctl restart yocr

systemd-status: ## 查看运行状态
	systemctl --no-pager status yocr | head -12

systemd-logs: ## 跟踪服务日志 (journalctl)
	journalctl -u yocr -f -n 50

systemd-uninstall: ## 卸载: 停止并删除 unit 文件
	sudo systemctl disable --now yocr 2>/dev/null; sudo rm -f $(SYSTEMD_UNIT); sudo systemctl daemon-reload
	@echo "已卸载"
