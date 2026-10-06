import os
import sys
import django

sys.path.insert(0, os.path.abspath('.'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'digitaltwin_project.settings')
django.setup()

from django.test import RequestFactory
from django.contrib.auth.models import User
from tracker.views import simulation_sandbox_view

rf = RequestFactory()
user = User.objects.get(id=4)
req = rf.get('/simulation/')
req.user = user

resp = simulation_sandbox_view(req)
print('HTTP status code:', resp.status_code)
assert resp.status_code == 200

content = resp.content.decode('utf-8')
print('Rendered template length (bytes):', len(content))

# Verify critical elements exist in rendered HTML
assert 'What-If Simulation Engine' in content
assert 'grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch' in content
assert 'lg:col-span-7 min-w-0' in content
assert 'lg:col-span-5 min-w-0' in content
assert 'slider-spend' in content
assert 'slider-side-income' in content
assert 'slider-shock' in content
assert 'slider-shock-day' in content
assert 'slider-inflation' in content
assert 'slider-study' in content
assert 'slider-sleep' in content
assert 'slider-work' in content
assert 'slider-screen' in content
assert 'toggle-sprint' in content
assert 'wealthChart' in content
assert 'focusChart' in content
assert 'crossImpactChart' in content
assert 'custom-savings-delta' in content
assert 'custom-focus-delta' in content
assert 'custom-burnout-risk' in content
assert 'relative h-72 w-full min-w-0' in content

# Verify no duplicate slider IDs
for slider_id in [
    'slider-spend', 'slider-side-income', 'slider-shock', 
    'slider-shock-day', 'slider-inflation', 'slider-study', 
    'slider-sleep', 'slider-work', 'slider-screen', 'toggle-sprint'
]:
    count = content.count(f'id="{slider_id}"')
    print(f'{slider_id} count: {count}')
    assert count == 1, f'Expected 1 instance of {slider_id}, got {count}'

# Verify no duplicate text or card IDs
assert content.count('id="wealthChart"') == 1
assert content.count('id="focusChart"') == 1
assert content.count('id="crossImpactChart"') == 1
assert content.count('id="btn-simulate"') == 1

# Test POST api_run_simulation to ensure payload and response align with JS logic
from tracker.views import api_run_simulation
import json

payload = {
    'expense_delta_pct': -10.0,
    'side_income_pct': 15.0,
    'shock_pct': 0.0,
    'shock_day': 30,
    'fixed_inflation_pct': 0.0,
    'study_hours_delta': 1.0,
    'sleep_target_hours': 7.5,
    'work_hours_delta': 0.0,
    'screen_limit_hours': 6.5,
    'sprint_mode': False
}
req_post = rf.post('/api/simulate/', data=json.dumps(payload), content_type='application/json')
req_post.user = user
resp_post = api_run_simulation(req_post)
assert resp_post.status_code == 200
data_post = json.loads(resp_post.content)
assert data_post['success'] is True
assert 'custom_impact' in data_post
assert 'trajectories' in data_post
assert 'deltas' in data_post
assert 'advisory_cards' in data_post
assert len(data_post['advisory_cards']) == 3

print('All verification checks (UI render + Async API) passed with flying colors!')
