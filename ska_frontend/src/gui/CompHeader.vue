<script setup lang="ts">
import { ServiceAuth } from '@/domain/auth/service_auth';
import { GlobalState } from '@/domain/global_state/global_state';
import { useGlobalStore } from '@/domain/global_state/global_store';

const globalState = GlobalState.instance();
const globalStore = useGlobalStore();


async function login() {
  await ServiceAuth.login();
}

async function logout() {
  await ServiceAuth.logout();

}

</script>

<template>
  <header class="navbar bg-neutral shadow-lg">
    <div class="flex flex-row flex-1">
      <label for="my-drawer" class="btn drawer-button">
        <img src="/assets/icons/bors-3.svg" width="24" />
      </label>
      <RouterLink :to="globalState.appRoutes.PAGE_HOME" class="btn btn-ghost text-neutral-content text-xl">
        Specific Knowledge Assistant
      </RouterLink>

      <span class="flex-1"></span>
      <details class="dropdown dropdown-end">
        <summary class="m-1 btn">
          <p>
            {{ globalStore.globalStore.userDetails != null ? globalStore.globalStore.userDetails.username : '' }}
          </p>
          <img src="/assets/icons/user-circle.svg" width="24" />
          <img src="/assets/icons/chevron-down.svg" width="16" />
        </summary>
        <ul class="p-2 shadow menu dropdown-content z-[1] bg-base-100 rounded-box w-32">
          <li v-if="globalStore.globalStore.userDetails == null">
            <button class="btn btn-ghost" @click="login">
              <img src="/assets/icons/arrow-right-start-on-rectangle.svg" width="24" />login
            </button>
          </li>
          <li v-if="globalStore.globalStore.userDetails != null">
            <button class="btn btn-ghost" @click="logout">
              <img src="/assets/icons/arrow-left-end-on-rectangle.svg" width="24" />logout
            </button>
          </li>
        </ul>
      </details>
    </div>
  </header>
</template>