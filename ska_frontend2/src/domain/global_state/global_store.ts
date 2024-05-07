import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { DtoUserDetails } from '../auth/dto_user_details'

export class GlobalStore {
  userDetails: DtoUserDetails | null;

  constructor(obj: {
    userDetails: DtoUserDetails | null,
  }) {
    this.userDetails = obj.userDetails;
  }
}

export const useGlobalStore = defineStore('globalStore', () => {
  const globalStore = ref(new GlobalStore({ userDetails: null }));

  return { globalStore }
})
