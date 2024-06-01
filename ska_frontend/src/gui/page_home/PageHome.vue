<script setup lang="ts">
import { GlobalState } from '@/domain/global_state/global_state';
import { useGlobalStore } from '@/domain/global_state/global_store';
import { ServiceAuth } from '@/domain/auth/service_auth';

const globalState = GlobalState.instance();
const globalStore = useGlobalStore();

async function login() {
  await ServiceAuth.login();
}
</script>

<template>
  <div class="ska-page-container">
    <div class="ska-page-column-flex">
      <div class="text-center flex flex-col justify-center justify-items-center content-center items-center">
        <img class="text-center justify-center justify-self-center" src="/assets/icons/chat-bubble-left-right.svg"
          width="64">
      </div>
      <div class="text-center">
        <h2>Welcome to Specific Knowledge Assistant</h2>
      </div>

      <div>
        <p>
          The Specific Knowldge Assistant (SKA) helps you find existing knowledge based on specific documents which have
          been given to the system by the admin.
        </p>
      </div>

      <template v-if="globalStore.globalStore.userDetails == null">
        <div>
          <p>Please <a @click="login">Login</a> in order to use SKA</p>
        </div>
      </template>
      <template v-else>
        <div>
          <p>
            Go to <RouterLink :to="globalState.appRoutes.PAGE_ASSISTANT">Assistant</RouterLink> page in order to use SKA
          </p>
        </div>
      </template>


    </div>
  </div>
</template>