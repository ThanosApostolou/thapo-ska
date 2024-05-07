<script setup lang="ts">
import { ServiceAuth } from '@/domain/auth/service_auth';
import { GlobalState } from '@/domain/global_state/global_state';
import { onMounted } from 'vue';
import { useRouter } from 'vue-router';

// state
const router = useRouter();

// lifecycles
onMounted(async () => {
  await afterLogin();
});

// functions
async function afterLogin() {
  await ServiceAuth.afterLogin(router);
}
</script>

<template>
  <div class="ska-page-container">
    <div class="ska-page-column-flex">
      <div>
        "please wait for redirect..."
      </div>
    </div>
  </div>
</template>

<style></style>