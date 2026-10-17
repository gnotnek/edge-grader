import { createContext, useContext, useEffect, useState } from 'react'

/** Bumped by the header's reload button; every useApi re-fetches when it changes. */
export const ReloadContext = createContext(0)

export interface ApiState<T> {
  data?: T
  error?: string
  loading: boolean
}

export function useApi<T>(path: string): ApiState<T> {
  const reload = useContext(ReloadContext)
  const [state, setState] = useState<ApiState<T>>({ loading: true })
  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true }))
    fetch(`/api${path}`)
      .then(async (r) => {
        const body = await r.json()
        if (!r.ok) throw new Error(body.error ?? r.statusText)
        return body as T
      })
      .then((data) => alive && setState({ data, loading: false }))
      .catch((e: Error) => alive && setState({ error: e.message, loading: false }))
    return () => {
      alive = false
    }
  }, [path, reload])
  return state
}
