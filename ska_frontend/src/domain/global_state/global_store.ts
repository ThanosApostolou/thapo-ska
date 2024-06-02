import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { DtoUserDetails } from '../auth/dto_user_details'

export class GlobalStore {
  userDetails: DtoUserDetails | null;
  isAppReady: boolean;

  constructor(obj: {
    userDetails: DtoUserDetails | null,
    isAppReady: boolean,
  }) {
    this.userDetails = obj.userDetails;
    this.isAppReady = obj.isAppReady;
  }
}

export const useGlobalStore = defineStore('globalStore', () => {
  const globalStore = ref(new GlobalStore({ userDetails: null, isAppReady: false }));

  async function waitAppReady(): Promise<void> {
    const globalStore = useGlobalStore();
    const mypromise = new Promise<void>((resolve, reject) => {
      if (globalStore.globalStore.isAppReady) {
        resolve();
      } else {
        const unsubscribe = globalStore.$subscribe((mutation, state) => {
          if (state.globalStore.isAppReady) {
            unsubscribe();
            resolve();
          }
        });
      }
    });
    return mypromise;

  }

  return { globalStore, waitAppReady }
})
