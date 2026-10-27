// The Streamlit front end. CD deploys this file on every release (it is idempotent),
// so the app is created the first time and updated with the new image after that.
param location string = resourceGroup().location

@description('Container Apps environment the API already runs in.')
param environmentId string

@description('Front end image, e.g. ghcr.io/owner/lanka-export-survival:ui-latest')
param image string

@description('Public URL of the prediction API.')
param apiUrl string

param appName string = 'ca-lanka-export-ui'

resource ui 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8501
        transport: 'auto' // Streamlit talks to the browser over a WebSocket
      }
    }
    template: {
      containers: [
        {
          name: 'ui'
          image: image
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'API_URL', value: apiUrl }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: { path: '/_stcore/health', port: 8501 }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1 // one replica keeps each browser session on the same server
      }
    }
  }
}
