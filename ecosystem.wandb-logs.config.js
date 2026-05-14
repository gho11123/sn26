module.exports = {
  apps: [
    {
      name: "perturb-wandb-logs",
      script: "download_run_logs.py",
      interpreter: "/home/client_6755_1/alex/Perturb/.venv/bin/python",
      args: "--watch",
      cwd: "/home/client_6755_1/alex/Perturb",
      autorestart: true,
      watch: false,
      max_restarts: 50,
      restart_delay: 5000,
      kill_timeout: 10000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "perturb-dashboard",
      script: "dashboard.py",
      interpreter: "/home/client_6755_1/alex/Perturb/.venv/bin/python",
      args: "--port 8800 --log wandb_logs/uid0.log",
      cwd: "/home/client_6755_1/alex/Perturb",
      autorestart: true,
      watch: false,
      max_restarts: 50,
      restart_delay: 3000,
      kill_timeout: 5000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};
