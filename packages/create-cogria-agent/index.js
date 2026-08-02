#!/usr/bin/env node
// create-cogria-agent — scaffold a minimal, runnable CogriaAgent project.
//
//   npx create-cogria-agent <dir> [--name <name>] [--framework-path <abs>] [--force]
//
// Generates a backend (business actions + kernel wiring), a docker-compose
// (agentserv + redis + optional postgres), and front-end wiring. Zero deps —
// just Node built-ins.

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync, chmodSync } from "node:fs";
import { dirname, join, resolve, basename } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TEMPLATE_DIR = join(__dirname, "templates", "default");

// Dotfiles can't be shipped literally in an npm package (npm strips some), so
// templates store them un-dotted and we restore the real name on copy.
const RENAME = {
  gitignore: ".gitignore",
  "env.example": ".env.example",
  "env.local.example": ".env.local.example",
};

function parseArgs(argv) {
  const out = { dir: null, name: null, frameworkPath: null, force: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--force") out.force = true;
    else if (a === "--name") out.name = argv[++i];
    else if (a === "--framework-path") out.frameworkPath = argv[++i];
    else if (a === "-h" || a === "--help") out.help = true;
    else if (!a.startsWith("-") && !out.dir) out.dir = a;
  }
  return out;
}

function usage() {
  console.log(`create-cogria-agent — scaffold a CogriaAgent project

Usage:
  npx create-cogria-agent <dir> [options]

Options:
  --name <name>            Project name (default: the target directory name)
  --framework-path <abs>   Absolute path to your CogriaAgent checkout
                           (auto-detected when run from inside the repo)
  --force                  Write into a non-empty directory
  -h, --help               Show this help
`);
}

// Find the framework root (the dir holding packages/agentserv) by walking up
// from this CLI's location.
function detectFrameworkPath() {
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    if (existsSync(join(dir, "packages", "agentserv", "pyproject.toml"))) return dir;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

function isTextLike() {
  return true; // every template file is UTF-8 text
}

function copyTree(srcDir, destDir, vars) {
  mkdirSync(destDir, { recursive: true });
  for (const entry of readdirSync(srcDir)) {
    const src = join(srcDir, entry);
    const renamed = RENAME[entry] || entry;
    const dest = join(destDir, renamed);
    if (statSync(src).isDirectory()) {
      copyTree(src, dest, vars);
    } else {
      let content = readFileSync(src, "utf8");
      for (const [k, v] of Object.entries(vars)) {
        content = content.split(k).join(v);
      }
      writeFileSync(dest, content);
      if (entry.endsWith(".sh")) chmodSync(dest, 0o755);
    }
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.dir) {
    usage();
    process.exit(args.dir ? 0 : 1);
  }

  const target = resolve(process.cwd(), args.dir);
  const projectName = args.name || basename(target);

  if (existsSync(target) && statSync(target).isDirectory() && readdirSync(target).length && !args.force) {
    console.error(`✖ ${target} is not empty. Use --force to scaffold into it anyway.`);
    process.exit(1);
  }

  let frameworkPath = args.frameworkPath || detectFrameworkPath();
  if (frameworkPath) {
    frameworkPath = resolve(frameworkPath);
    if (!existsSync(join(frameworkPath, "packages", "agentserv", "pyproject.toml"))) {
      console.error(`✖ --framework-path ${frameworkPath} doesn't look like a CogriaAgent checkout (no packages/agentserv).`);
      process.exit(1);
    }
  } else {
    frameworkPath = "/path/to/CogriaAgent";
    console.warn(
      "⚠ Could not auto-detect the CogriaAgent framework path. Set it later in\n" +
        "  agent/pyproject.toml and .env (search for /path/to/CogriaAgent), or re-run\n" +
        "  with --framework-path. (Not needed once cogria-* are published to PyPI.)\n"
    );
  }

  const vars = {
    __PROJECT_NAME__: projectName,
    __FRAMEWORK_PATH__: frameworkPath,
  };

  copyTree(TEMPLATE_DIR, target, vars);

  const rel = args.dir;
  console.log(`✓ Scaffolded "${projectName}" into ${target}\n`);
  console.log("Next steps:");
  console.log(`  cd ${rel}/agent`);
  console.log("  cp .env.example .env        # set OPENAI_* / AGENT_MODEL / JWT_SECRET");
  console.log("  uv run --with pytest --with pytest-asyncio pytest tests   # smoke test");
  console.log("  set -a; source .env; set +a");
  console.log("  uv run uvicorn app.app:app --host 127.0.0.1 --port 8001\n");
  console.log("Or with Docker (kernel + redis):");
  console.log(`  cd ${rel} && cp .env.example .env && docker compose up\n`);
  console.log("Then run the front-end — see the generated README.md.");
}

main();
