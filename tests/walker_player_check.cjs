// Run the actual embedded player script with a minimal canvas and DOM surface.
// Every mount represents a new iframe after a notebook learning change.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

for (const player of JSON.parse(fs.readFileSync(0, 'utf8'))) {
    const elements = new Map();
    const context = new Proxy({}, {get: (target, name) => target[name] ?? (() => {})});
    const callbacks = [];
    function element(id) {
        if (!elements.has(id)) elements.set(id, {
            textContent: '', value: id === 'speed' ? '1' : '0',
            clientWidth: 900, width: 0, height: 0,
            getContext: () => context,
            setAttribute: () => {},
        });
        return elements.get(id);
    }
    const sandbox = {
        document: {getElementById: element},
        window: {devicePixelRatio: 1, addEventListener: () => {}},
        requestAnimationFrame: callback => callbacks.push(callback),
    };
    vm.runInNewContext(player.html.match(/<script>([\s\S]*?)<\/script>/)[1], sandbox);
    assert.equal(element('policy').textContent, player.label);
    if (player.duration === 0) {
        assert.equal(element('play').disabled, true);
        assert.equal(element('seek').disabled, true);
        continue;
    }
    callbacks.shift()(0);
    callbacks.shift()(100);
    if (!player.autoplay) {
        assert.equal(Number(element('seek').value), 0, 'Opting out must keep new rollouts paused');
        element('play').onclick();
        callbacks.shift()(200);
        callbacks.shift()(300);
    }
    assert.ok(Number(element('seek').value) > 0,
        'A new learning outcome must start animating instead of remaining at time zero');
    assert.ok(element('action').textContent.includes(player.alpha),
        'The visible action must come from this learning outcome');
    assert.equal(element('play').textContent, 'Pause');
    element('play').onclick();
    const pausedTime = Number(element('seek').value);
    callbacks.shift()(400);
    assert.equal(Number(element('seek').value), pausedTime, 'Manual pause must still work');
    element('restart').onclick();
    callbacks.shift()(500);
    callbacks.shift()(600);
    assert.ok(Number(element('seek').value) > 0, 'Replay must restart the current policy');
    callbacks.shift()(1000 * (player.duration + 1));
    assert.equal(Number(element('seek').value), player.duration);
    callbacks.shift()(1000 * (player.duration + 2.3));
    if (player.autoplay) {
        assert.equal(Number(element('seek').value), 0, 'Hold the outcome, then repeat the updated policy');
    } else {
        assert.equal(element('play').textContent, 'Play', 'Manual playback must stop at the end');
    }
}
