PYTHON ?= python3
MISE_MUJOCO_GL ?= osmesa
EPISODES ?= 20
PLANNER_SAMPLES ?= 10000
LABEL_SAMPLES ?= 1000
MODEL_DIR ?= artifacts/models/contact_act_local
TRAIN_STEPS ?= 6000

.PHONY: install setup scene demo contact full test eval eval-headless eval-recovery bench submission-video submission-check data train distill labels video clean web console replay export planner-data check-plan

setup: install

web:
	cd web && npm ci && npm run build

console: scene web
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m mise.cli serve

replay: web
	$(PYTHON) -m mise.cli serve --read-only

install:
	$(PYTHON) -m pip install -c constraints.txt -e '.[test]'

scene:
	$(PYTHON) scripts/build_scene.py

demo: scene
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m mise.cli demo --video artifacts/day1_drawer_expert.mp4

contact:
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m mise.cli contact

full:
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m mise.cli full

test: scene
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m unittest discover -s tests -v

eval: scene
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m mise.cli eval --seeds eval/seeds.yaml

eval-headless:
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) scripts/evaluate_full_task.py --seeds eval/seeds.yaml

eval-recovery:
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) scripts/evaluate_recovery_comparison.py --seeds eval/seeds.yaml

bench:
	$(PYTHON) bench/intel_bench.py --device CPU --precision f32

submission-video:
	$(PYTHON) scripts/build_submission_video.py

submission-check: test
	cd web && npm run build
	$(PYTHON) scripts/build_submission_video.py --check-only
	@echo "Local software and retained evidence checks passed. Intel target and Docker-engine checks remain machine-specific."

check-plan:
	$(PYTHON) -m mise.cli check-plan

# Fixture data is for integration only; it does not demonstrate learned grasps.
data: scene
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m mise.cli collect --episodes $(EPISODES)

planner-data: scene
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m mise.cli planner-data --count $(PLANNER_SAMPLES)

train:
	$(PYTHON) scripts/train_contact.py --output $(MODEL_DIR) --steps $(TRAIN_STEPS)

export:
	$(PYTHON) scripts/export_contact.py --checkpoint $(MODEL_DIR)/checkpoint.pt --output $(MODEL_DIR)/openvino

distill:
	@echo "Deferred in the revised brief; prioritize ACT export and verified task completion."
	@exit 1

labels: scene
	MUJOCO_GL=$(MISE_MUJOCO_GL) $(PYTHON) -m mise.cli label-data --samples $(LABEL_SAMPLES)

video: demo

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; shutil.rmtree('artifacts', ignore_errors=True)"
