"""Episode capture statistics independent of rewards and boundary penalties."""
import csv
import json


class TagEpisodeMetrics:
    """Remember each prey's first collision; later separation is not escape."""

    def __init__(self, world):
        self.first_capture = {a.name: None for a in world.agents if not a.adversary}
        if not self.first_capture:
            raise ValueError('Capture metrics require at least one good agent')

    def update(self, world, scenario, step):
        if step < 1:
            raise ValueError('Capture steps start at 1')
        adversaries = scenario.adversaries(world)
        for prey in scenario.good_agents(world):
            if self.first_capture[prey.name] is None and prey.collide and any(
                    adv.collide and scenario.is_collision(prey, adv) for adv in adversaries):
                self.first_capture[prey.name] = step

    def result(self, episode, steps):
        captured = [s for s in self.first_capture.values() if s is not None]
        success = len(captured) == len(self.first_capture)
        return {
            'episode': episode,
            'steps': steps,
            'capture_success': success,
            'capture_step': max(captured) if success else None,
            'escaped_agents': len(self.first_capture) - len(captured),
            'total_good_agents': len(self.first_capture),
            'first_capture_steps': dict(self.first_capture),
        }


def summarize_tag_metrics(records):
    successful = [r['capture_step'] for r in records if r['capture_success']]
    total_good = sum(r['total_good_agents'] for r in records)
    return {
        'episodes': len(records),
        'successful_capture_episodes': len(successful),
        'adversary_capture_success_rate': len(successful) / len(records) if records else None,
        'average_capture_steps': sum(successful) / len(successful) if successful else None,
        'agent_escape_success_rate': (
            sum(r['escaped_agents'] for r in records) / total_good if total_good else None),
    }


def save_tag_metrics(output_dir, records, settings):
    """Save completed episodes and aggregate rates (0..1); no capture -> null mean."""
    fields = ['episode', 'steps', 'capture_success', 'capture_step',
              'escaped_agents', 'total_good_agents', 'first_capture_steps']
    with (output_dir / 'evaluation_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, 'first_capture_steps': json.dumps(record['first_capture_steps'])})
    summary = {**summarize_tag_metrics(records), 'settings': settings}
    (output_dir / 'evaluation_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary
