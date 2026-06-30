const API_BASE = '/api';

export async function detectObjects(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 min timeout

    try {
        const response = await fetch(`${API_BASE}/detect`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            if (response.status === 503) {
                throw new Error('Model initialization failed. Please check that model weights exist and refresh.');
            }
            throw new Error(errorData.detail || `Detection failed (HTTP ${response.status})`);
        }

        return await response.json();
    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === 'AbortError') {
            throw new Error('Detection timed out (3 min). Models may still be loading — please try again.');
        }
        if (err.message?.includes('Failed to fetch') || err.message?.includes('NetworkError') || err.name === 'TypeError') {
            throw new Error(
                'Cannot connect to backend server. Start it with: cd frontend/backend && python server.py'
            );
        }
        throw err;
    }
}

export async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
        if (!response.ok) throw new Error('Backend unavailable');
        return await response.json();
    } catch {
        return { status: 'offline', models: {} };
    }
}

export async function getModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/models`, { signal: AbortSignal.timeout(5000) });
        if (!response.ok) throw new Error('Failed to fetch model info');
        return await response.json();
    } catch {
        return null;
    }
}
