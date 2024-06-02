<script setup lang="ts">
import { ServiceAuth } from '@/domain/auth/service_auth';
import RootPage from './PageRoot.vue';
import { onMounted } from 'vue';
import { useGlobalStore } from '@/domain/global_state/global_store';
import { useRouter } from 'vue-router';

// hooks
const router = useRouter();
const globalStore = useGlobalStore();

// lifecycles
onMounted(async () => {
  await initialize();
});

// functions
async function initialize() {
  await ServiceAuth.initialAuth(globalStore.globalStore, router);
}

</script>

<template>
  <div v-if="!globalStore.globalStore.isAppReady" indeterminate color="primary">loading...</div>
  <RootPage v-if="globalStore.globalStore.isAppReady" />

</template>