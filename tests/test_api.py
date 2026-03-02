def test_execute_masc_endpoint_validation_error(test_client):
    # Missing required fields
    response = test_client.post("/v1/masc/execute", json={"task_description": "Do something"})
    assert response.status_code == 422


def test_execute_masc_max_adversaries_limit(test_client, base_masc_config):
    # Duplicate adversaries to exceed limit (default max is 5)
    for i in range(10):
        base_masc_config.adversaries.append(
            base_masc_config.adversaries[0]
        )

    payload = {
        "task_description": "Test high adversaries",
        "config": base_masc_config.model_dump()
    }

    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 400
    assert "Exceeded max adversaries" in response.json()["detail"]